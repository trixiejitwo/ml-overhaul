"""Small shared helpers used across the forecasting package."""
import pandas as pd


def trailing_window(s: pd.Series, days: int) -> pd.Series:
    """Return the trailing `days`-day slice of a datetime-indexed series.
    Equivalent to the deprecated/removed pandas Series.last('{days}D')."""
    if len(s) == 0:
        return s
    cutoff = s.index.max() - pd.Timedelta(days=days)
    return s.loc[s.index > cutoff]
