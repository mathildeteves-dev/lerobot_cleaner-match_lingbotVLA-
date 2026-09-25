"""Legacy R6 compatibility facade. The pipeline uses separate checks/transforms.

Direct callers retain apply(work), which performs both stages in sequence.
"""
from collections import Counter

from lerobot_cleaner.v21.rules.base import BaseRule
from lerobot_cleaner.v21.rules.numeric import numeric_checks, numeric_transforms


class NumericSanityRule(BaseRule):
    name = "numeric_sanity"

    def __init__(self, config, dataset, outlier_bounds=None):
        super().__init__(config, dataset)
        self.checks = numeric_checks(config, dataset, outlier_bounds)
        self.transforms = numeric_transforms(config, dataset, outlier_bounds)

    def run(self, work, context=None):
        for rule in self.checks + self.transforms:
            if work.dropped:
                break
            rule.stats = Counter()
            rule.run(work, context)
            self.stats.update(rule.stats)
