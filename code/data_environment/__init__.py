"""Data environment setup pipeline modules."""

__all__ = ["setup_steps"]

def setup_steps() -> list[str]:
    """Return the ordered data environment setup scripts."""
    return [
        "code.data_environment.01_prep_data",
        "code.data_environment.02_solve_equilibrium",
        "code.data_environment.03_draw_workers",
    ]
