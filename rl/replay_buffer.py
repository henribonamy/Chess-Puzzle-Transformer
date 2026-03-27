from collections import deque


class ReplayBuffer:
    """Stores qualifying positions for novelty filtering using exact hash lookups."""

    def __init__(self, max_size: int) -> None:
        self._seeds: set[str] = set()
        self._discovered_set: set[str] = set()
        self._discovered_deque: deque[str] = deque(maxlen=max_size)

    def seed(self, board_str: str) -> None:
        self._seeds.add(board_str)

    def add(self, board_str: str, pv: str) -> None:
        if len(self._discovered_deque) == self._discovered_deque.maxlen:
            evicted = self._discovered_deque[0]
            self._discovered_set.discard(evicted)
        self._discovered_deque.append(board_str)
        self._discovered_set.add(board_str)

    def is_novel(self, board_str: str, pv: str = "", tau_board: int = 0, tau_pv: int = 0) -> bool:
        return board_str not in self._seeds and board_str not in self._discovered_set

    def __len__(self) -> int:
        return len(self._seeds) + len(self._discovered_set)
