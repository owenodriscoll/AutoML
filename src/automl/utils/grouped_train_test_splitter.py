from sklearn.model_selection import train_test_split


def groupwise_train_test_split(
    X,
    y,
    group_names: list[str],
    test_size: float = 0.2,
    shuffle=True,
    random_state=None,
):
    """
    Splits the dataset into train and test sets while ensuring that all rows 
    belonging to the same group (as defined by `group_names`) are kept together 
    in either the train or test set. This is basically a multi-column stratification.

    Parameters:
        X (pd.DataFrame): Feature dataset with a multi-index.
        y (pd.DataFrame or pd.Series): Target dataset with a multi-index.
        group_names (list[str]): List of column names that define the grouping.
        test_size (float): Proportion of the dataset to include in the test split.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training features.
            - X_test (pd.DataFrame): Testing features.
            - y_train (pd.DataFrame or pd.Series): Training targets.
            - y_test (pd.DataFrame or pd.Series): Testing targets.

    Raises:
        AssertionError: If the index names of X and y do not match.
        AssertionError: If there are overlapping indexes between train and test sets.

    Example:
        >>> import pandas as pd
        >>> from grouped_train_test_splitter import groupwise_train_test_split
        >>> data = {
        ...     'group': ['A', 'A', 'B', 'B', 'C', 'C'],
        ...     'feature': [1, 2, 3, 4, 5, 6],
        ...     'target': [0, 1, 0, 1, 0, 1]
        ... }
        >>> df = pd.DataFrame(data).set_index(['group', df.index])
        >>> X = df[['feature']]
        >>> y = df['target']
        >>> X_train, X_test, y_train, y_test = groupwise_train_test_split(
        ...     X, y, group_names=['group'], test_size=0.5, random_state=42
        ... )
        >>> print(X_train)
        >>> print(X_test)
    """

    index_names = X.index.names
    assert list(X.index.names) == list(
        y.index.names
    ), "Index names need to be identical for X and y"

    X_reset = X.reset_index()
    y_reset = y.reset_index()

    unique_pairs = X_reset[group_names].drop_duplicates()

    train_pairs, test_pairs = train_test_split(
        unique_pairs, test_size=test_size, shuffle=shuffle, random_state=random_state
    )

    X_train = X_reset.merge(train_pairs, on=group_names)
    X_test = X_reset.merge(test_pairs, on=group_names)
    y_train = y_reset.merge(train_pairs, on=group_names)
    y_test = y_reset.merge(test_pairs, on=group_names)

    X_train.set_index(index_names, inplace=True)
    X_test.set_index(index_names, inplace=True)
    y_train.set_index(index_names, inplace=True)
    y_test.set_index(index_names, inplace=True)

    overlap_X = X_train.index.intersection(X_test.index)
    overlap_y = y_train.index.intersection(y_test.index)
    assert (overlap_X.empty) & (
        overlap_y.empty
    ), "Overlapping indexes found in Train and Test datasets"

    return X_train, X_test, y_train, y_test
