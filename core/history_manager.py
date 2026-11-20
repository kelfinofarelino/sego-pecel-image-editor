import numpy as np


class HistoryManager:
    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self.original_image: np.ndarray | None = None
        self.current_image: np.ndarray | None = None
        self.current_label: str = "Original Image"
        self.undo_stack: list[tuple[np.ndarray, str]] = []
        self.redo_stack: list[tuple[np.ndarray, str]] = []

    def load_image(self, image: np.ndarray, label: str = "Original Image") -> None:
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.current_label = label
        self.undo_stack.clear()
        self.redo_stack.clear()

    def push_state(self, new_image: np.ndarray, label: str = "Applied Effect") -> None:
        if self.current_image is not None:
            self.undo_stack.append((self.current_image.copy(), self.current_label))
            if len(self.undo_stack) > self.max_steps:
                self.undo_stack.pop(0)
        self.current_image = new_image.copy()
        self.current_label = label
        self.redo_stack.clear()

    def undo(self) -> np.ndarray | None:
        if not self.undo_stack or self.current_image is None:
            return None
        self.redo_stack.append((self.current_image.copy(), self.current_label))
        self.current_image, self.current_label = self.undo_stack.pop()
        return self.current_image.copy()

    def redo(self) -> np.ndarray | None:
        if not self.redo_stack or self.current_image is None:
            return None
        self.undo_stack.append((self.current_image.copy(), self.current_label))
        self.current_image, self.current_label = self.redo_stack.pop()
        return self.current_image.copy()

    def reset(self) -> np.ndarray | None:
        if self.original_image is None:
            return None
        if self.current_image is not None:
            self.undo_stack.append((self.current_image.copy(), self.current_label))
        self.current_image = self.original_image.copy()
        self.current_label = "Reset to Original"
        self.redo_stack.clear()
        return self.current_image.copy()

    def get_history_entries(self) -> tuple[list[str], int]:
        if self.original_image is None:
            return [], -1
        past_labels = [lbl for _, lbl in self.undo_stack]
        future_labels = [lbl for _, lbl in reversed(self.redo_stack)]
        entries = past_labels + [self.current_label] + future_labels
        active_index = len(past_labels)
        return entries, active_index

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0