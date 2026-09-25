"""Episode inclusion is an explicit transform/writer decision, never file deletion."""


class EpisodeFilter:
    @staticmethod
    def select_plans(plans):
        return [plan for plan in plans if not plan.reject_episode]

    @staticmethod
    def keep(episode, plan):
        if episode.episode_index != plan.episode_id:
            raise ValueError("Plan refers to another episode")
        if plan.reject_episode:
            episode.drop("; ".join(plan.reasons) or "rejected by policy")
            return False
        return True
