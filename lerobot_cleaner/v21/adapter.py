"""Compatibility bridge to the version-independent episode contract."""


class V21Adapter:
    @staticmethod
    def to_trajectory(episode, fps):
        from lerobot_cleaner.adapters.episode import UnifiedEpisode
        if isinstance(episode, UnifiedEpisode):
            return episode.to_trajectory(fps)
        return UnifiedEpisode(ref=getattr(episode, "ref", None), df=episode.df,
                              keep_indices=episode.keep_indices).to_trajectory(fps)

    @staticmethod
    def resolve_target(resolver, dotted):
        resolved = resolver.resolve(dotted)
        return resolved.modality, slice(resolved.start, resolved.end)
