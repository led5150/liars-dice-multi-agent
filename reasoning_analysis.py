# reasoning_analysis.py

import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any, Union
import os
from openai import OpenAI
from pydantic import BaseModel
import textwrap


class ClusterSummary(BaseModel):
    """Pydantic model for structured output from OpenAI"""
    category_name: str
    decision_type: str
    reasoning_pattern: str
    key_characteristics: List[str]
    confidence_score: float

class ReasoningAnalyzer:
    """
    Analyzes LLM reasoning patterns by generating embeddings, calculating similarities,
    clustering similar reasonings, summarizing clusters, and rendering visualizations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ReasoningAnalyzer with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters for analysis
        """
        self.config = config
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config['embedding_model'])
        
        # Set n_neighbors based on dataset size
        n_neighbors = 2 if len(config.get('text_data', [])) < 50 else 15
        
        # Initialize UMAP for dimensionality reduction
        self.umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            min_dist=0.1,
            metric='cosine',
            random_state=42  # For reproducibility
        )
        
        # Initialize clustering parameters
        self.max_clusters = config.get('clustering', {}).get('max_clusters', 8)
        self.min_cluster_size = config.get('clustering', {}).get('min_cluster_size', 5)
        self.min_samples = config.get('clustering', {}).get('min_samples', 3)
        
        # Store processed data
        self.embeddings = None
        self.similarities = None
        self.clusters = None
        self.summaries = None

    def generate_embeddings(self, text_data: List[str]) -> np.ndarray:
        """
        Generate embeddings for the given text data using sentence transformers.
        
        Args:
            text_data: List of text strings to generate embeddings for
            
        Returns:
            numpy.ndarray: Matrix of embeddings
        """
        # Generate embeddings
        self.embeddings = self.embedding_model.encode(text_data)
        return self.embeddings

    def calculate_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embeddings: Matrix of embeddings to calculate similarities between
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        # Normalize embeddings
        normalized = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        # Calculate cosine similarity
        self.similarities = np.dot(normalized, normalized.T)
        
        return self.similarities

    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """
        Find optimal number of clusters using cluster stability with less conservative thresholds.
        
        Args:
            data: The data to cluster
            
        Returns:
            Optimal number of clusters
        """
        max_clusters = min(self.max_clusters, len(data) // 3)  # Allow more clusters
        if max_clusters < 2:
            return 2

        # Try different min_cluster_sizes to find stable clustering
        best_score = -1
        best_size = max(3, len(data) // 12)  # Start with smaller clusters
        
        # Test different min_cluster_sizes with smaller steps
        step_size = max(1, len(data) // 100)  # Smaller steps for finer control
        for min_size in range(3, len(data) // 3, step_size):
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_size,
                min_samples=max(1, min_size // 2),  # More aggressive min_samples
                metric='euclidean',
                cluster_selection_method='eom',
                cluster_selection_epsilon=0.5  # More lenient cluster selection
            )
            clusterer.fit(data)
            
            # Skip if we get too many or too few clusters
            n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
            if n_clusters < 2 or n_clusters > max_clusters:
                continue
            
            # Calculate weighted score based on both stability and number of clusters
            stabilities = []
            cluster_sizes = []
            for label in set(clusterer.labels_):
                if label != -1:  # Skip noise points
                    cluster_mask = clusterer.labels_ == label
                    stability = np.mean(clusterer.probabilities_[cluster_mask])
                    size = np.sum(cluster_mask)
                    stabilities.append(stability)
                    cluster_sizes.append(size)
            
            if stabilities:
                avg_stability = np.mean(stabilities)
                # Favor solutions with more clusters if stability is decent
                cluster_bonus = n_clusters / max_clusters
                size_penalty = max(cluster_sizes) / len(data)  # Penalize very large clusters
                
                score = avg_stability * (1 + cluster_bonus * 0.5) * (1 - size_penalty * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_size = min_size
        
        print(f"Found optimal min_cluster_size: {best_size}")
        return best_size

    def cluster_reasonings(self, similarities: np.ndarray) -> Dict[str, Any]:
        """
        Cluster similar reasonings using HDBSCAN.
        
        Args:
            similarities: Similarity matrix between reasonings
            
        Returns:
            Dict containing cluster labels and reduced dimensionality data for visualization
        """
        # Reduce dimensionality for clustering and visualization
        reduced_data = self.umap_reducer.fit_transform(self.embeddings)
        
        # Use simpler clustering approach with fixed parameters
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(3, len(reduced_data) // 10),
            min_samples=3,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        cluster_labels = self.clusterer.fit_predict(reduced_data)
        
        self.clusters = {
            'labels': cluster_labels,
            'reduced_data': reduced_data,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        }
        
        return self.clusters

    def _get_cluster_examples(self, cluster_indices: np.ndarray, text_data: List[str], 
                            embeddings: np.ndarray, n_examples: int = 10) -> List[str]:
        """
        Get the most representative examples from a cluster based on distance to centroid.
        
        Args:
            cluster_indices: Indices of points in the cluster
            text_data: Original text data
            embeddings: Embeddings of the text data
            n_examples: Number of examples to return
            
        Returns:
            List of the most representative reasoning examples
        """
        # Get cluster embeddings
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calculate distances to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # Get indices of n closest points to centroid
        closest_indices = np.argsort(distances)[:n_examples]
        
        # Get the actual examples
        examples = [text_data[cluster_indices[i]] for i in closest_indices]
        
        return examples

    def _summarize_cluster_llm(self, examples: List[str]) -> ClusterSummary:
        """
        Use GPT-4 to summarize a cluster based on representative examples.
        
        Args:
            examples: List of representative reasoning examples from the cluster
            
        Returns:
            ClusterSummary object containing the analysis
        """
        # Create the prompt
        prompt = f"""You are analyzing reasonings from players in a game of Liar's Dice. 
        Each reasoning explains why a player made a particular move.
        
        Here are {len(examples)} representative reasonings from a cluster:
        
        {chr(10).join(f"{i+1}. {example}" for i, example in enumerate(examples))}
        
        Analyze these reasonings and categorize them based on the common decision-making patterns.
        Consider aspects like:
        - Is this a aggressive or conservative strategy?
        - What type of move is being made (bluff, safe play, strategic deception)?
        - What factors are the players considering?
        - How confident are the players in their decisions?
        """
        
        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are analyzing player decision-making patterns in Liar's Dice."},
                    {"role": "user", "content": prompt}
                ],
                response_format=ClusterSummary,
            )
            
            return completion.choices[0].message.parsed
            
        except Exception as e:
            print(f"Error in LLM summarization: {e}")
            # Return a basic summary as fallback
            return ClusterSummary(
                category_name="Error in Analysis",
                decision_type="Unknown",
                reasoning_pattern="Could not analyze pattern",
                key_characteristics=["Error in processing"],
                confidence_score=0.0
            )

    def summarize_clusters(self, clusters: Dict[str, Any], text_data: List[str]) -> Dict[int, ClusterSummary]:
        """
        Summarize each cluster using LLM-based analysis of representative examples.
        
        Args:
            clusters: Dictionary containing cluster labels and reduced data
            text_data: Original text data that was clustered
            
        Returns:
            Dict mapping cluster IDs to their summaries
        """
        labels = clusters['labels']
        unique_labels = np.unique(labels)
        summaries = {}
        
        for label in unique_labels:
            if label != -1:  # Skip noise points
                # Get indices of points in this cluster
                cluster_indices = np.where(labels == label)[0]
                
                # Get representative examples
                examples = self._get_cluster_examples(
                    cluster_indices, 
                    text_data, 
                    self.embeddings, 
                    n_examples=10
                )
                
                # Get LLM summary
                summary = self._summarize_cluster_llm(examples)
                summaries[label] = summary
        
        self.summaries = summaries
        return summaries

    def render_visualizations(self, summaries: Dict[int, ClusterSummary], text_data: List[str], output_dir: str, filename_suffix: str = '') -> None:
        """
        Render visualizations for the clustered decision types.
        
        Args:
            summaries: Dict mapping cluster IDs to their summaries
            text_data: Original text data that was clustered
            output_dir: Directory to save visualization files
            filename_suffix: Optional suffix to append to output filenames
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create scatter plot of clusters
        reduced_data = self.clusters['reduced_data']
        labels = self.clusters['labels']
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'cluster': labels,
            'text': text_data
        })
        
        # Create scatter plot
        scatter_fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['text'],
            title='Reasoning Clusters',
            labels={'cluster': 'Strategy'}
        )
        
        # Create a mapping for cluster labels
        cluster_labels = {str(i): f"Cluster {i}" for i in df['cluster'].unique()}
        cluster_labels['-1'] = 'Mixed'  # Explicitly set -1 to "Mixed"
        
        # Update traces for all clusters
        for cluster_id in df['cluster'].unique():
            scatter_fig.update_traces(
                selector=dict(name=str(cluster_id)),
                name=cluster_labels[str(cluster_id)]
            )
        
        # Add cluster labels as annotations
        for cluster_id in df['cluster'].unique():
            cluster_points = df[df['cluster'] == cluster_id]
            if len(cluster_points) > 0:  # Only add annotation if cluster has points
                center_x = cluster_points['x'].mean()
                center_y = cluster_points['y'].mean()
                label = cluster_labels[str(cluster_id)]
                
                scatter_fig.add_annotation(
                    x=center_x,
                    y=center_y,
                    text=label,
                    showarrow=True,
                    arrowhead=1,
                    font=dict(size=12)
                )
        
        # Update scatter plot layout
        scatter_fig.update_layout(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            showlegend=True,
            title_x=0.5
        )
        
        # Create table figure
        # Transpose table_data and process each column separately
        table_headers = ['Cluster', 'Category', 'Decision Type', 'Pattern', 'Key Characteristics', 'Confidence']
        table_data = []
        
        for cluster_id, summary in summaries.items():
            if cluster_id != -1:
                table_data.append([
                    f"Cluster {cluster_id}",
                    summary.category_name,
                    summary.decision_type,
                    summary.reasoning_pattern,
                    '<br>'.join(f"• {textwrap.fill(char, width=40)}" for char in summary.key_characteristics),
                    f"{summary.confidence_score:.2f}"
                ])
        
        transposed_data = list(zip(*table_data))
        processed_columns = []
        
        for i, (header, column) in enumerate(zip(table_headers, transposed_data)):
            if header == 'Key Characteristics':
                processed_cells = []
                for cell in column:
                    # Split into individual points
                    points = [p.strip() for p in cell.split('•') if p.strip()]
                    # Format each point with a dash and wrap text
                    formatted_points = []
                    for point in points:
                        # Wrap each point to 50 characters max
                        wrapped_text = textwrap.fill(point, width=50, subsequent_indent='    ')
                        wrapped_text = wrapped_text.replace('\n', '<br>')
                        formatted_points.append(f"- {wrapped_text}")
                    # Join points with newlines
                    formatted_text = '\n'.join(formatted_points)
                    processed_cells.append(formatted_text)
                processed_columns.append(processed_cells)
            else:
                # For other columns, wrap long text as well
                processed_cells = []
                for cell in column:
                    if len(str(cell)) > 50:  # Only wrap if text is long
                        wrapped = textwrap.fill(str(cell), width=50)
                        processed_cells.append(wrapped)
                    else:
                        processed_cells.append(str(cell))
                processed_columns.append(processed_cells)

        table_fig = go.Figure(data=[go.Table(
            header=dict(
                values=table_headers,
                align='left',
                font=dict(size=12),
                height=40,
                line_color='darkslategray',
                fill=dict(color='lightgrey')
            ),
            cells=dict(
                values=processed_columns,
                align='left',
                font=dict(size=11),
                height=None,  # Auto-adjust height based on content
                line_color='darkslategray',
                fill=dict(color=['white', 'white'])
            ),
            columnwidth=[0.6, 0.8, 0.8, 1.0, 1.3, 0.5]  # Give more width to Key Characteristics
        )])

        # Update layout to ensure proper text wrapping
        table_fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            width=1400,  # Increased width further
            height=None  # Auto-adjust height
        )
        
        # Save visualizations in specified formats
        for fmt in self.config['visualization']['output_formats']:
            if fmt == 'html':
                with open(f"{output_dir}/reasoning_clusters{filename_suffix}.html", 'w') as f:
                    f.write(scatter_fig.to_html(full_html=False))
                    f.write(table_fig.to_html(full_html=False))
            elif fmt == 'png':
                scatter_fig.write_image(f"{output_dir}/reasoning_clusters{filename_suffix}.png")
                # For PNG, adjust table size to ensure all content is visible
                table_fig.update_layout(
                    width=1500,  # Wider for PNG
                    height=150 * (len(table_data) + 1),  # Dynamic height based on number of rows
                    margin=dict(l=50, r=50, t=50, b=50)  # Larger margins for PNG
                )
                table_fig.write_image(f"{output_dir}/cluster_summaries{filename_suffix}.png")
