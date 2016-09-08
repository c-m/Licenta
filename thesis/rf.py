importances = clf.feature_importances_
# Calculate the std dev of each feature
std = np.std([t.feature_importances_ for t in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
idx_names = list()
for i in range(X_train.shape[1]):
    idx_names.append(data['feature_names'][indices[i]])

# Print the feature ranking
print('Feature ranking:')
for f in range(X_train.shape[1]):
    print('%d. feature %d, name: %s, (%f)' % \
         (f + 1, indices[f], data['feature_names'][indices[f]], 
          importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title('Feature importances')
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), idx_names)
plt.xlim([-1, X_train.shape[1]])
plt.show()
