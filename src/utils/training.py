from typing import List, Union
import torch.nn as nn


def freeze_components(model: nn.Module, components_to_freeze: Union[List[str], str]):
    """
    モデルの指定されたコンポーネントのパラメータを凍結します。

    Args:
        model (nn.Module): 凍結するコンポーネントを含むモデル。
        components_to_freeze (Union[List[str], str]): 凍結するコンポーネント名のリスト、または単一のコンポーネント名。
                                                      例: ["adapter", "vitakka", "vicara"]
    """
    if isinstance(components_to_freeze, str):
        components_to_freeze = [components_to_freeze]

    for name, module in model.named_children():
        if name in components_to_freeze:
            print(f"Freezing component: {name}")
            for param in module.parameters():
                param.requires_grad = False
        else:
            print(f"Keeping component unfrozen: {name}")
            for param in module.parameters():
                param.requires_grad = True


def unfreeze_components(model: nn.Module, components_to_unfreeze: Union[List[str], str]):
    """
    モデルの指定されたコンポーネントのパラメータを解凍します。

    Args:
        model (nn.Module): 解凍するコンポーネントを含むモデル。
        components_to_unfreeze (Union[List[str], str]): 解凍するコンポーネント名のリスト、または単一のコンポーネント名。
    """
    if isinstance(components_to_unfreeze, str):
        components_to_unfreeze = [components_to_unfreeze]

    for name, module in model.named_children():
        if name in components_to_unfreeze:
            print(f"Unfreezing component: {name}")
            for param in module.parameters():
                param.requires_grad = True


def check_frozen_status(model: nn.Module):
    """
    モデルの各コンポーネントの凍結状態（requires_grad）を表示します。
    """
    print("\n--- Component Freezing Status ---")
    for name, module in model.named_children():
        all_frozen = True
        all_unfrozen = True
        for i, param in enumerate(module.parameters()):
            if i == 0:  # Check only the first parameter to represent component status
                if param.requires_grad:
                    all_frozen = False
                else:
                    all_unfrozen = False
            # If there are mixed states within a module, it's more complex.
            # For simplicity, we assume all params in a module are either frozen or unfrozen by `freeze_components`.

        if not list(module.parameters()):  # Handle modules with no parameters
            print(f"Component '{name}': No trainable parameters.")
        elif all_frozen:
            print(f"Component '{name}': FROZEN")
        elif all_unfrozen:
            print(f"Component '{name}': UNLOCKED")
        else:
            print(f"Component '{name}': MIXED STATE (investigate manually)")
    print("-------------------------------")
