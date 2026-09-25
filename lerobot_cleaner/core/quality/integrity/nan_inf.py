"""NaN/Inf check; retain the public finite result key for compatibility."""
from ..finite import check_finite

check_nan_inf = check_finite
