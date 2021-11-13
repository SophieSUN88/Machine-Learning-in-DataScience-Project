# K-means clustering
# put the funtions together again with  fit and mse_classes

def initialize_centroids(data, k =2):
    Centroids =[]
    for _ in range(k):
        i = np.random.randint(len(data))
        Centroids.append(X[i,:])
    return Centroids

    
def assignment(x,Centroids):
    distances = np.array([distance(x,centroid) for centroid in Centroids])
    return np.argmin(distances)

def make_classes(data, Centroids):
    A = dict()
    for i in range(len(Centroids)):
        A[i]=[]
    for x in data:
        A[assignment(x,Centroids)].append(x)
    return A

def new_centroids(data, Centroids):
    new_centroids = []
    A = make_classes(data, Centroids)
    for i in range(len(Centroids)):
        new_centroids.append((1/len(A[i]))*sum(A[i]))
    return new_centroids

def fit(data, k =3 , max_iterations=100, epsilon = 0.01):
    C_old = initialize_centroids(data,k=k)
    C_new = new_centroids(data,C_old)
    centroid_distances = [distance(p[0], p[1]) for p in zip(C_old, C_new)]
    iterations = 0

    while max(centroid_distances)> epsilon and iterations< max_iterations:
        C_old, C_new = C_new, new_centroids(data,C_new)
        centroid_distances = [distance(p[0], p[1]) for p in zip(C_old, C_new)]
        iterations +=1
    return C_new

def mse_classes(data, Centroids):
    errors = []
    A_classes = make_classes(data, Centroids)
    for i, centroid in enumerate(Centroids):
        error = sum(0.5*(centroid - a) @ (centroid - a) for a in A_classes[i])
        errors.append(error)    
    return sum(x for x in errors)


# example
# import some data to play with
iris = datasets.load_iris()
X = iris.data
# we only need the first 2 columns
X= X[:,:2]
X.shape

C1 = fit(X,k=1)
C2 = fit(X,k=2)
C3 = fit(X,k=3)
C4 = fit(X,k=4)
C5 = fit(X,k=5)
C6 = fit(X,k=6)
C=[C1, C2,C3,C4,C5,C6]

errors = [mse_classes(X,centroids) for centroids in C]
# plot the progress of the improvement in errors 
errors = [mse_classes(X,centroids) for centroids in C]
plt.figure(figsize=(10,6))
plt.plot([i for i in range(1,7)],errors)

# plot the data
for x in X:
    if assignment(x,C6) == 0:
        plt.scatter(x[0],x[1],color ="blue")
    elif assignment(x,C6) == 1:
        plt.scatter(x[0],x[1],color ="green")
    elif assignment(x,C6) == 2:
        plt.scatter(x[0],x[1],color ="yellow")
    elif assignment(x,C6) == 3:
        plt.scatter(x[0],x[1],color ="purple")
    elif assignment(x,C6) == 4:
        plt.scatter(x[0],x[1],color ="black")
    else:
        plt.scatter(x[0],x[1],color ="red")