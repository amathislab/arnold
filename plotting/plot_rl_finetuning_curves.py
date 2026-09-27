import _srcpath  # noqa: F401
from plot_learning_curves import main

if __name__ == "__main__":
    main(default_panel="finetuning", default_relative=False, default_window=101, default_smoothing="savgol", default_combine=True)
