# Hasil variasi eksperimen

Letakkan setiap variasi dalam **subfolder sendiri** yang berisi file yang sama seperti `Main_model/Multi_Model/artifacts/`:

- `patchtst_suhu.pth`, `preprocessor_suhu.joblib`
- `patchtst_humidity.pth`, `preprocessor_humidity.joblib`
- … (lima regresi + `patchtst_cuaca.pth`, `preprocessor_cuaca.joblib`)

`prediction_forecast.csv` saja **tidak cukup** untuk evaluasi ini (eval membutuhkan bobot model).

## Skrip evaluasi

`eval_artifacts.py` meniru **`Main_model/eval_exp.py`**: metrik regresi pada saluran target terstandarisasi, akurasi per langkah untuk cuaca, pemisahan validasi dari **permutasi indeks jendela** dengan `--train-fraction` dan `--seed` (default 42).

Jalankan dari **akar proyek** `ModelTA`:

```bash
# Contoh: 80/20, horizon 720
python results/eval_artifacts.py --artifacts-dir results/720_80-20 --pred-len 720 --train-fraction 0.8

# 80/20, horizon 336
python results/eval_artifacts.py --artifacts-dir results/336-80-20 --pred-len 336 --train-fraction 0.8

# 70/30, horizon 720
python results/eval_artifacts.py --artifacts-dir results/720_70-30 --pred-len 720 --train-fraction 0.7

# 70/30, horizon 336
python results/eval_artifacts.py --artifacts-dir results/336-70-30 --pred-len 336 --train-fraction 0.7
```

Sesuaikan nama folder dengan struktur Anda. `--pred-len` **wajib** cocok dengan checkpoint (336 vs 720).

## Catatan metodologi

- Pelatihan memakai `torch.randperm`; skrip ini memakai **`numpy.random.default_rng(seed).permutation`** seperti `eval_exp.py`, jadi himpunan validasi **tidak identik** bit-per-bit dengan split saat training, tetapi **rasio** dan **seed** dapat diselaraskan untuk pelaporan.
- Regresi: preprocessor di-eval **di-fit ulang** pada seluruh `historical_environment` (sama seperti `eval_exp.py`), bukan memuat `preprocessor_*.joblib` dari folder artefak. Cuaca: preprocessor **dimuat** dari disk.
