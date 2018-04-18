# Random_Forest

### Part 1 - Description of the theory





### Part 2 - Basic tests on simulated data



In [ ]: from sklearn.datasets import make_blobs
x_blobs,y_blobs = make_blobs(n_samples=2000,n_features=2,centers=4,random_state=0)


from matplotlib import pyplot
pyplot.scatter(x[:,0],x[:,1],c=y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(x_train,y_train)

print("Accuracy on the training set {}%".format(classifier.score(x_train,y_train)*100))
print("Accuracy on the test set {}%".format(classifier.score(x_test,y_test)*100))



train_acc = list()
test_acc = list() # list to add the test set accuracies
test_ks = range(1,25)# list containing values of k to be tested

for k in tqdm.tqdm(test_ks):
    local_classifier = RandomForestClassifier(n_estimators=k)
    local_classifier.fit(x_train,y_train)
    train_acc.append(local_classifier.score(x_train,y_train))
    test_acc.append(local_classifier.score(x_test,y_test))

plt.plot(test_ks,train_acc,color="blue",label="train set")
plt.plot(test_ks,test_acc,color="green",label="test set")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend()
print("Best k: {}, Best test accuracy {}%".format(test_ks[np.argmax(test_acc)],max(test_acc)*100))

from sklearn.metrics import classification_report,confusion_matrix
y_pred_train = classifier.predict(x_train)
report = classification_report(y_true=y_train,y_pred=y_pred_train)
matrix = confusion_matrix(y_true=y_train,y_pred=y_pred_train)
print("Training Set:")
print(report)
print(matrix)
plt.matshow(matrix)
plt.colorbar()
plt.xlabel("Real class")
plt.ylabel("Predicted class")

y_pred_test = classifier.predict(x_test)
report = classification_report(y_true=y_test,y_pred=y_pred_test)
matrix = confusion_matrix(y_true=y_test,y_pred=y_pred_test)
print("Test Set:")
print(report)
print(matrix)
plt.matshow(matrix)
plt.colorbar()
plt.xlabel("Real class")
plt.ylabel("Predicted class")


from matplotlib.colors import ListedColormap
def plot_boundaries(classifier,X,Y,h=0.2):
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x0, x1 = np.meshgrid(np.arange(x0_min, x0_max,h),
                         np.arange(x1_min, x1_max,h))
    dataset = np.c_[x0.ravel(),x1.ravel()]
    Z = classifier.predict(dataset)

    # Put the result into a color plot
    Z = Z.reshape(x0.shape)
    plt.figure()
    plt.pcolormesh(x0, x1, Z)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y,
                edgecolor='k', s=20)
    plt.xlim(x0.min(), x0.max())
    plt.ylim(x1.min(), x1.max())
plot_boundaries(classifier,x_train,y_train)



### Part 3 - Advanced tests and analysis on Pyrat datasets
