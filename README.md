# Student Scores Analysis with NumPy

A small data analysis project that uses **NumPy** to explore simulated exam scores for 30 students across 5 subjects (Maths, English, Kiswahili, Science, SST). It covers array statistics, boolean masking, z-score standardisation, grading with `np.where`, and correlation.

## What the script does

1. Generates a reproducible 30 x 5 score matrix (`default_rng(42)`, scores 30-99)
2. Reports shape, dtype and overall mean
3. Computes the mean per subject and total per student
4. Finds the best student and the strongest subject
5. Counts students scoring above 80 in Maths and the share of the class
6. Standardises each subject with z-scores and checks the column means are ~0
7. Adds an average column and assigns grades (A 75+, B 60-74, C 50-59, D below 50)
8. Measures the Maths vs Science correlation with `np.corrcoef`

## Sample results (seed 42)

| Metric | Result |
|---|---|
| Overall mean | 65.57 |
| Highest-mean subject | SST (69.40) |
| Best student | Index 9 (total 411) |
| Above 80 in Maths | 11 students (36.67%) |
| Grades | A: 5, B: 18, C: 6, D: 1 |
| Maths vs Science correlation | 0.18 (weak) |

The weak correlation is expected: the scores are random and independent, so no real relationship exists between subjects.

## How to run

```bash
git clone https://github.com/<your-username>/students-score-analysis.git
cd students-score-analysis
pip install -r requirements.txt
python students_project.py
```

## Skills demonstrated

NumPy arrays, axis-wise aggregation, boolean indexing, broadcasting, `np.where`, `np.unique`, `np.corrcoef`

## Author

Patrick Njuguna Muchemi
