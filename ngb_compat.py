"""NGBoost compatibility fix, kept in its own file so the saved model can be loaded by the app."""
from ngboost import NGBRegressor


class NGBoostRegressor(NGBRegressor):
    """NGBRegressor that tells scikit-learn when it has been fitted (needed inside a Pipeline)."""

    def __sklearn_is_fitted__(self):
        return hasattr(self, "base_models") and len(self.base_models) > 0
