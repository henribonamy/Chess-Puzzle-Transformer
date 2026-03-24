from collections import deque


def levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


class ReplayBuffer:
    """Stores qualifying positions for novelty filtering.

    Seeds (training dataset positions) use fast exact-match lookups.
    Discovered positions (RL-generated) use Levenshtein distance checks.
    """

    def __init__(self, max_size: int) -> None:
        """Initialise with a fixed-size deque for discovered positions and a seed set."""
        self._seeds: set[str] = set()
        self._discovered: deque[tuple[str, str]] = deque(maxlen=max_size)

    def seed(self, board_str: str) -> None:
        """Add a training dataset position to the seed set."""
        self._seeds.add(board_str)

    def add(self, board_str: str, pv: str) -> None:
        """Append an RL-discovered qualifying position."""
        self._discovered.append((board_str, pv))

    def is_novel(self, board_str: str, pv: str, tau_board: int, tau_pv: int, sample_size: int = 200) -> bool:
        """Return True if position is not in seeds and sufficiently distant from a random sample of discovered."""
        if board_str in self._seeds:
            return False
        discovered = list(self._discovered)
        if len(discovered) > sample_size:
            import random
            discovered = random.sample(discovered, sample_size)
        for buf_board, buf_pv in discovered:
            if levenshtein_distance(board_str, buf_board) < tau_board:
                return False
            if tau_pv > 0 and levenshtein_distance(pv, buf_pv) < tau_pv:
                return False
        return True

    def __len__(self) -> int:
        """Return total number of entries (seeds + discovered)."""
        return len(self._seeds) + len(self._discovered)
