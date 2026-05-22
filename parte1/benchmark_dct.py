import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fft import dctn


# DCT 1D fatta a mano usando la formula vista a lezione
def dct1d_naive(f):
    N = len(f)
    k = np.arange(N)
    alpha = np.sqrt(1.0 / N) * np.ones(N)
    alpha[1:] = np.sqrt(2.0 / N)
    indices = np.arange(N)
    return alpha * np.sum(
        f * np.cos(np.pi * k[:, np.newaxis] * (2 * indices + 1) / (2 * N)), axis=1
    )


# DCT 2D: prima faccio la DCT 1D su ogni riga e poi su ogni colonna
def dct2d_naive(block):
    N = block.shape[0]
    temp = np.zeros((N, N))
    # DCT sulle righe
    for i in range(N):
        temp[i, :] = dct1d_naive(block[i, :])
    result = np.zeros((N, N))
    # DCT sulle colonne
    for j in range(N):
        result[:, j] = dct1d_naive(temp[:, j])
    return result


# controllo del caso test dato nel testo del progetto
def verifica_caso_test():
    block_8x8 = np.array(
        [
            [231, 32, 233, 161, 24, 71, 140, 245],
            [247, 40, 248, 245, 124, 204, 36, 107],
            [234, 202, 245, 167, 9, 217, 239, 173],
            [193, 190, 100, 167, 43, 180, 8, 70],
            [11, 24, 210, 177, 81, 243, 8, 112],
            [97, 195, 203, 47, 125, 114, 165, 181],
            [193, 70, 174, 167, 41, 30, 127, 245],
            [87, 149, 57, 192, 65, 129, 178, 228],
        ],
        dtype=float,
    )

    SEP = "=" * 68

    print(SEP)
    print("  VERIFICA CASO TEST")
    print(SEP)

    # confronto sulla prima riga del blocco
    attesa_1d = np.array(
        [4.01e02, 6.60e00, 1.09e02, -1.12e02, 6.54e01, 1.21e02, 1.16e02, 2.88e01]
    )
    calcolata_1d = dct1d_naive(block_8x8[0, :])
    errore_1d = np.max(np.abs(calcolata_1d - attesa_1d))

    print("\n  DCT 1D — prima riga del blocco 8×8\n")
    header = f"  {'k':>3}  {'Attesa':>12}  {'Calcolata':>12}  {'Errore':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in range(8):
        print(
            f"  {k:>3}  {attesa_1d[k]:>12.2f}  {calcolata_1d[k]:>12.2f}  {abs(calcolata_1d[k]-attesa_1d[k]):>10.4f}"
        )
    print(f"\n  Errore massimo: {errore_1d:.4f}  ", end="")
    print("✓ OK" if errore_1d < 1.0 else "✗ ATTENZIONE")

    # confronto tra DCT 2D calcolata con la nostra funzione e target teorico
    calcolata_2d = dct2d_naive(block_8x8)
    attesa_2d = np.array(
        [
            [1.11e03, 4.40e01, 7.59e01, -1.38e02, 3.50e00, 1.22e02, 1.95e02, -1.01e02],
            [7.71e01, 1.14e02, -2.18e01, 4.13e01, 8.77e00, 9.90e01, 1.38e02, 1.09e01],
            [4.48e01, -6.27e01, 1.11e02, -7.63e01, 1.24e02, 9.55e01, -3.98e01, 5.85e01],
            [
                -6.99e01,
                -4.02e01,
                -2.34e01,
                -7.67e01,
                2.66e01,
                -3.68e01,
                6.61e01,
                1.25e02,
            ],
            [
                -1.09e02,
                -4.33e01,
                -5.55e01,
                8.17e00,
                3.02e01,
                -2.86e01,
                2.44e00,
                -9.41e01,
            ],
            [-5.38e00, 5.66e01, 1.73e02, -3.54e01, 3.23e01, 3.34e01, -5.81e01, 1.90e01],
            [
                7.88e01,
                -6.45e01,
                1.18e02,
                -1.50e01,
                -1.37e02,
                -3.06e01,
                -1.05e02,
                3.98e01,
            ],
            [
                1.97e01,
                -7.81e01,
                9.72e-01,
                -7.23e01,
                -2.15e01,
                8.13e01,
                6.37e01,
                5.90e00,
            ],
        ],
        dtype=float,
    )
    errore_2d = np.max(np.abs(calcolata_2d - attesa_2d))

    print(f"\n  DCT 2D — blocco 8×8\n")
    print(f"  Elemento [0,0]  atteso:    1.11e+03")
    print(f"  Elemento [0,0]  calcolato: {calcolata_2d[0,0]:.2e}")
    print(f"  Errore max vs target teorico: {errore_2d:.2e}  ", end="")
    print("✓ OK" if errore_2d < 15.0 else "✗ ATTENZIONE")

    print(f"\n{SEP}\n")


# funzione per confrontare i tempi della versione naive e della libreria
def benchmark(N_values, ripetizioni=3):
    tempi_naive = []
    tempi_libreria = []

    SEP = "=" * 68
    print(SEP)
    print("  BENCHMARK")
    print(SEP)
    print(f"\n  {'N':>5}  {'Naïve [s]':>14}  {'Scipy [s]':>14}  {'Speedup':>10}")
    print("  " + "-" * 50)

    for N in N_values:
        M = np.random.rand(N, N)

        # tempo versione naive
        t = []
        for _ in range(ripetizioni):
            t0 = time.perf_counter()
            dct2d_naive(M)
            t.append(time.perf_counter() - t0)
        t_naive = min(t)

        # tempo versione scipy
        t = []
        for _ in range(ripetizioni):
            t0 = time.perf_counter()
            dctn(M, norm="ortho")
            t.append(time.perf_counter() - t0)
        t_lib = min(t)

        speedup = t_naive / t_lib if t_lib > 0 else float("inf")
        print(f"  {N:>5}  {t_naive:>14.5f}  {t_lib:>14.6f}  {speedup:>9.1f}x")

        tempi_naive.append(t_naive)
        tempi_libreria.append(t_lib)

    print()
    return np.array(tempi_naive), np.array(tempi_libreria)


# grafico finale con scala logaritmica sulle ordinate
def plot_benchmark(N_values, tempi_naive, tempi_libreria):
    N_arr = np.array(N_values, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.semilogy(
        N_arr,
        tempi_naive,
        "o-",
        color="steelblue",
        linewidth=2,
        label="DCT2 naïve  (atteso: O(N³))",
    )
    ax.semilogy(
        N_arr,
        tempi_libreria,
        "s-",
        color="tomato",
        linewidth=2,
        label="DCT2 scipy  (atteso: O(N²logN))",
    )

    # curve teoriche di riferimento
    ref_N3 = tempi_naive[0] * (N_arr / N_arr[0]) ** 3
    ref_N2lN = (
        tempi_libreria[0]
        * (N_arr**2 * np.log(N_arr))
        / (N_arr[0] ** 2 * np.log(N_arr[0]))
    )

    ax.semilogy(
        N_arr, ref_N3, "--", color="steelblue", alpha=0.4, label="Riferimento O(N³)"
    )
    ax.semilogy(
        N_arr, ref_N2lN, "--", color="tomato", alpha=0.4, label="Riferimento O(N²logN)"
    )

    ax.set_xlabel("N  (dimensione matrice N×N)", fontsize=12)
    ax.set_ylabel("Tempo [s]  —  scala logaritmica", fontsize=12)
    ax.set_title("Benchmark DCT2: implementazione naïve vs scipy (FFT)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("parte1/benchmark_dct.png", dpi=150)
    print("Grafico salvato: benchmark_dct.png")
    plt.show()


if __name__ == "__main__":
    verifica_caso_test()

    N_values = [4, 8, 16, 32, 64, 100]
    tempi_naive, tempi_libreria = benchmark(N_values, ripetizioni=3)
    plot_benchmark(N_values, tempi_naive, tempi_libreria)
