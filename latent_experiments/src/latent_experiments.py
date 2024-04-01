import pandas as pd
import numpy as np
import sklearn.metrics.pairwise
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    Normalizer,
    MaxAbsScaler,
    PowerTransformer,
)
# import seaborn as sns


__all__ = ["calculate_scaled_cosine_similarity", "split_data_on_column"]


def calculate_scaled_cosine_similarity(
    data: pd.DataFrame, scale_method: str = "minmax"
) -> pd.DataFrame:
    """
    Calculate the scaled cosine similarity matrix for the given data.

    Parameters:
    - data: The input data to calculate the cosine similarity matrix for.
    - scale_method: The method to scale the data. Default is 'minmax'.

    Returns:
    - cosine_similarity_df: The cosine similarity matrix, with the maximum similarity index for each row.

    Raises:
    - ValueError: If the specified scale method is not recognized.
    - ValueError: If the input data is empty.

    """

    if data.empty:
        raise ValueError(
            "The input data is empty. Please provide a non-empty DataFrame."
        )

    # Scale the data according to the specified method
    scalers = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "l2": Normalizer(norm="l2"),
        "l1": Normalizer(norm="l1"),
        "maxabs": MaxAbsScaler(),
        "yeojohnson": PowerTransformer(method="yeo-johnson"),
    }

    scaler = scalers.get(scale_method.lower())

    if not scaler:
        raise ValueError(
            f"Scaling method '{scale_method}' is not recognized. "
            "Choose from 'minmax', 'standard', 'robust', 'l2', 'l1', 'maxabs', or 'yeojohnson'."
        )

    normalized_data = scaler.fit_transform(data)

    normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

    cosine_similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(normalized_df)

    # Set diagonal to 0 from 1
    np.fill_diagonal(cosine_similarity_matrix, 0)

    # Convert to DataFrame and add max_index column
    cosine_similarity_df = pd.DataFrame(
        cosine_similarity_matrix, index=normalized_df.index, columns=normalized_df.index
    )
    cosine_similarity_df["max_index"] = cosine_similarity_df.idxmax(axis=1)

    return cosine_similarity_df


def split_data_on_column(df, column_name, gap=0.9):
    """
    Split the data in a DataFrame based on a specified column.

    Args:
        df (pandas.DataFrame): The DataFrame to split.
        column_name (str): The name of the column to split on.
        gap (float, optional): The percentage of data to exclude from both ends. Defaults to 0.9.

    Returns:
        tuple: A tuple containing the lower subset and upper subset of the data.

    """

    if df.empty:
        raise ValueError(
            "The input data is empty. Please provide a non-empty DataFrame."
        )

    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")

    gap_length = (1 - gap) / 2
    lower_quantile = df[column_name].quantile(0 + gap_length)
    upper_quantile = df[column_name].quantile(1 - gap_length)

    print(lower_quantile)
    print(upper_quantile)

    lower_subset = df[df[column_name] <= lower_quantile]
    middle_subset = df[
        (df[column_name] > lower_quantile) & (df[column_name] <= upper_quantile)
    ]
    upper_subset = df[df[column_name] >= upper_quantile]

    return lower_subset, upper_subset


def match_by_column(
    data_df, treatment_column, outcome_column, observation_column, gap_size=0.9
):

    a, b = split_data_on_column(data_df, treatment_column, gap_size)

    df1_vector = a[observation_column].to_numpy()
    df2_vector = b[observation_column].to_numpy()

    similarity_matrix = cosine_similarity(df1_vector, df2_vector)

    most_similar_idx = np.argmax(similarity_matrix, axis=1)
    most_similar_rows = b.iloc[most_similar_idx]

    result = pd.concat(
        [a.reset_index(drop=True), most_similar_rows.reset_index(drop=True)], axis=1
    )

    data_list = []
    for i in range(len(result)):
        headline_1, headline_2 = result.iloc[i][observation_column]
        formal_1, formal_2 = result.iloc[i][treatment_column]
        ctr_1, ctr_2 = result.iloc[i][outcome_column]

        data_list.append(
            (headline_1, headline_2, formal_1, formal_2, ctr_1, ctr_2, ctr_1 - ctr_2)
        )

    # Convert list into DataFrame
    final_df = pd.DataFrame(
        data_list,
        columns=[
            observation_column + "_1",
            observation_column + "_2",
            "treatment_1",
            "treatment_2",
            outcome_column + "_1",
            outcome_column + "_2",
            "difference",
        ],
    )
    final_df["type"] = treatment_column
    final_df["gap"] = gap_size
    return final_df


def match_by_cosine_similarity(data, treatment_var_name, outcome_var_name, gap=0.9):

    # Split into upper and lower quantile
    higher, lower = split_data_on_column(
        data.drop(outcome_var_name, axis=1), treatment_var_name, gap=gap
    )

    # Drop the column we are measuring so we can match on other columns
    df1_vector = higher.drop(treatment_var_name, axis=1)
    df2_vector = lower.drop(treatment_var_name, axis=1)

    # Create the similarity matrix
    cosine_similarity_matrix = cosine_similarity(df1_vector, df2_vector)
    cosine_similarity_df = pd.DataFrame(
        cosine_similarity_matrix, index=df1_vector.index, columns=df2_vector.index
    )

    # Find the mathces
    most_similar_indices = cosine_similarity_df.idxmax(axis=1)

    #
    matched_rows_a = higher.loc[most_similar_indices.index].reset_index(drop=True)
    matched_rows_b = lower.loc[most_similar_indices.values].reset_index(drop=True)

    matched_rows_a = matched_rows_a.add_suffix("_a")
    matched_rows_b = matched_rows_b.add_suffix("_b")

    matched_df = pd.concat([matched_rows_a, matched_rows_b], axis=1)

    a_outcome = data.loc[most_similar_indices.index, outcome_var_name].reset_index(
        drop=True
    )
    b_outcome = data.loc[most_similar_indices.values, outcome_var_name].reset_index(
        drop=True
    )

    matched_df["Outcome_a"] = a_outcome
    matched_df["Outcome_b"] = b_outcome

    matched_df["Outcome_diff"] = matched_df["Outcome_b"] - matched_df["Outcome_a"]

    return matched_df


def run_latent_experiments(
    data,
    columns_to_match,
    sns_theme="white",
    figsize=(15, 10),
    outcome_var="Outcome",
    match_threshold=0.2,
    draw_plot=True,
):
    """
    Plots outcome differences using cosine similarity matching across specified columns.

    Args:
    data (pandas.DataFrame): The input data with columns to be matched and outcome variable.
    columns_to_match (list): The columns to be used for matching.
    sns_theme (str): The Seaborn theme to apply to the plot.
    figsize (tuple): The size of the figure.
    outcome_var (str): The name of the outcome variable in the data.
    match_threshold (float): The cosine similarity threshold for matching.

    Returns:
    None: Displays the plot.
    """
    # Set the Seaborn theme for the plot
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # This DataFrame will collect all 'Outcome_diff' distributions for each group
    all_outcome_diffs = pd.DataFrame()

    for column in columns_to_match:
        matched_data = match_by_cosine_similarity(
            data, column, outcome_var, match_threshold
        )

        # Apply the hyperbolic arcsine transformation to the 'Outcome_diff' column
        matched_data["Outcome_diff"] = np.arcsinh(matched_data["Outcome_diff"])

        # Add a group column to distinguish the distributions
        matched_data["Group"] = column
        all_outcome_diffs = pd.concat(
            [all_outcome_diffs, matched_data[["Outcome_diff", "Group"]]]
        )

    # Convert the 'Group' column to a categorical type to preserve order in the plot
    all_outcome_diffs["Group"] = pd.Categorical(
        all_outcome_diffs["Group"], categories=columns_to_match
    )

    # # Define the color palette for the plot
    # if draw_plot:
    #     pal = sns.cubehelix_palette(len(columns_to_match), rot=-0.25, light=0.7)

    #     # Set up the FacetGrid
    #     g = sns.FacetGrid(
    #         all_outcome_diffs,
    #         row="Group",
    #         hue="Group",
    #         aspect=15,
    #         height=1.0,
    #         palette=pal,
    #     )

    #     # Draw the densities without fill color and transparent lines
    #     g.map(
    #         sns.kdeplot,
    #         "Outcome_diff",
    #         bw_adjust=0.5,
    #         clip_on=False,
    #         fill=True,
    #         alpha=1,
    #         linewidth=1.5,
    #     )
    #     g.map(
    #         sns.kdeplot, "Outcome_diff", clip_on=False, color="w", lw=2, bw_adjust=0.5
    #     )

    #     # Draw a vertical line for the average of each distribution
    #     def draw_average_line(x, **kwargs):
    #         ax = plt.gca()
    #         mean = np.mean(x)
    #         ax.axvline(
    #             mean, color="black", lw=2, ls="-"
    #         )  # Customise color, linewidth, and linestyle as needed

    #     g.map(draw_average_line, "Outcome_diff")

    #     # Define a simple function to label the plot in axes coordinates
    #     def label(x, color, label):
    #         ax = plt.gca()
    #         ax.text(
    #             0,
    #             0.2,
    #             label,
    #             fontweight="bold",
    #             color=color,
    #             ha="left",
    #             va="center",
    #             transform=ax.transAxes,
    #         )

    #     g.map(label, "Outcome_diff")

    #     # Set the subplots to overlap
    #     g.figure.subplots_adjust(hspace=-0.25)

    #     # Remove axes details that don't play well with overlap
    #     g.set_titles("")
    #     g.set(yticks=[], ylabel="")
    #     g.despine(bottom=True, left=True)

    #     current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #     plt.savefig(f"latent_exp_{current_date}.pdf")

    return all_outcome_diffs
