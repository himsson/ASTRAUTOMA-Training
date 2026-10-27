"""Нейросетевое ядро KIA: PPO на PyTorch.

    device.py       выбор GPU/CPU и диагностика
    spaces.py       пространство наблюдений (сырая телеметрия) и действий
    networks.py     actor-critic, нормализация входов
    ppo.py          алгоритм PPO, буфер, сохранение весов в kia_model.pth
    curriculum.py   учебная программа: три ступени и переносы весов
    sim_env.py      ступень 1: быстрый тренажёр выведения
    sim_env_plus.py ступень 2: скрытые механики живой KSP
    sim_env_pro.py  ступень 3: 3D-баллистика, таблицы атмосферы Кербина
    builder_env.py  среда сборки ракеты из палитры деталей игры
    flight_env.py   живая KSP как среда обучения с подкреплением
    trainer.py      циклы обучения и оценки

Ступени тренажёра наследуются одна от другой и обязаны выдавать
ОДИНАКОВЫЙ вектор наблюдения (40 чисел). Иначе веса прошлой ступени не
загрузятся, и вся идея постепенного усложнения рассыплется.
"""
from .device import DeviceInfo, get_device, inspect

__all__ = ["DeviceInfo", "get_device", "inspect", "is_available",
           "build_brain", "NeuralTrainer", "PPOConfig"]


def is_available() -> bool:
    """Установлен ли PyTorch (без него нейромодуль не грузится)."""
    from .device import TORCH_AVAILABLE
    return TORCH_AVAILABLE


def __getattr__(name):
    """Ленивая загрузка тяжёлых частей: pyTorch тянется только при обращении."""
    if name == "build_brain":
        from .ppo import build_brain
        return build_brain
    if name == "NeuralTrainer":
        from .trainer import NeuralTrainer
        return NeuralTrainer
    if name == "PPOConfig":
        from .ppo import PPOConfig
        return PPOConfig
    raise AttributeError(name)
