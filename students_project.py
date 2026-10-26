"""Student scores analysis with NumPy.

Simulates scores for 30 students across 5 subjects and answers a set of
exercise questions using vectorised NumPy operations.
"""

import numpy as np

rng = np.random.default_rng(42)
subjects = ["Maths", "English", "Kiswahili", "Science", "SST"]
scores = rng.integers(30, 100, size=(30, 5))  # 30 students x 5 subjects
n_students = scores.shape[0]

# 1. Shape, dtype and overall mean
print("Shape:", scores.shape)
print("Dtype:", scores.dtype)
print("Overall mean:", scores.mean().round(2))

# 2. Mean score per subject (axis=0) and total score per student (axis=1)
subject_means = scores.mean(axis=0)
student_totals = scores.sum(axis=1)
print("\nMean per subject:", {s: float(m) for s, m in zip(subjects, subject_means.round(2))})
print("Total per student:", student_totals)

# 3. Best student (highest total) and subject with the highest mean
best_student = student_totals.argmax()
best_subject = subjects[subject_means.argmax()]
print(f"\nBest student: index {best_student} (total {student_totals[best_student]})")
print("Subject with highest mean:", best_subject)

# 4. Students scoring above 80 in Maths
maths = scores[:, 0]
above_80 = (maths > 80).sum()
print(f"\nAbove 80 in Maths: {above_80} students "
      f"({above_80 / n_students * 100:.2f}% of the class)")

# 5. Standardise each subject with z-scores: (x - mean) / std
z = (scores - scores.mean(axis=0)) / scores.std(axis=0)
print("\nZ-score column means (should be ~0):", z.mean(axis=0).round(6))

# 6. Add an average column, then assign grades with np.where
students_avg = scores.mean(axis=1)
scores_with_avg = np.column_stack([scores, students_avg])  # 30 x 6

grades = np.where(students_avg >= 75, "A",
         np.where(students_avg >= 60, "B",
         np.where(students_avg >= 50, "C", "D")))

labels, counts = np.unique(grades, return_counts=True)
print("\nGrade distribution:", {str(l): int(c) for l, c in zip(labels, counts)})

# 7. Correlation between Maths and Science
r = np.corrcoef(scores[:, 0], scores[:, 3])[0, 1]
print("\nMaths vs Science correlation:", round(r, 2))
# r is close to 0 (weak positive): performance in Maths says very little about
# Science here. Expected, since the scores are random and independent.
