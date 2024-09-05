# SHAP


SHAP (Shapley Additive Explanations) is a method for explaining individual predictions of machine learning models using Shapley values from cooperative game theory. Shapley values assign a contribution to each feature based on how much it contributes to the model's prediction, averaged over all possible subsets of features.


For each feature j, the Shapley value ϕ_j is computed as an average over all possible subsets S of features that do not contain j. The formula is:
phi_j = sum_{S subset N \ {j}} [ ( |S|! * (n - |S| - 1)! ) / n! ] * ( f(S U {j}) - f(S) )
Where:
phi_j is the Shapley value for feature j.
S is a subset of the full feature set N that does not include j.
|S| is the number of features in subset S.
f(S) is the model’s prediction when only the features in S are used.
f(S U {j}) is the model’s prediction when feature j is added to subset S.
n is the total number of features.


f(S U {j}) - f(S) is the marginal contribution of feature j to the prediction. It shows how the prediction changes when j is added to the subset S.

Weighting Factor: The term ( |S|! * (n - |S| - 1)! ) / n!
is the weight that balances how many ways we can form the subset S and the complement of S with respect to the feature set N. It ensures that the Shapley value is an average over all possible subsets of features.
|S|! is the factorial of the number of features in subset S.
(n - |S| - 1)! is the factorial of the number of remaining features (excluding j).
n! is the factorial of the total number of features.


Efficiency: The sum of the Shapley values for all features equals the total difference between the model's prediction for the instance and the baseline (the average prediction over all instances).
sum_{j=1}^{n} phi_j = f(x) - f_baseline

Symmetry: If two features contribute equally to all subsets, they receive the same Shapley value.

Dummy: If a feature doesn’t change the prediction for any subset, its Shapley value is zero.

Additivity: For models that are combinations of sub-models, the Shapley values of the combined model are the sum of the Shapley values of the individual models.


In machine learning, SHAP values explain the prediction of a model for a single instance. The prediction is explained as the sum of the SHAP values for each feature:
f(x) = f_baseline + sum_{j=1}^{n} phi_j
Where:
f(x) is the model’s prediction for instance x.
f_baseline is the average model prediction across all instances.
phi_j is the Shapley value for feature j, indicating its contribution to the prediction.


Exact calculation of Shapley values requires evaluating the model on all subsets of features, which is computationally expensive (2^n subsets). The SHAP algorithm uses an approximation with a kernel function to efficiently compute Shapley values.

The Shapley kernel used for approximation is:
w(S) = (n - 1) / [ C(n, |S|) * |S| * (n - |S|) ]
Where:
w(S) is the weight for the subset S.
C(n, |S|) is the binomial coefficient, the number of ways to choose |S| features from n features.

