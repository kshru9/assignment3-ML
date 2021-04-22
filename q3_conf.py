
def visualise_confusion(y_hat,y,digit):
    mat = [[0,0],[0,0]]
    samples = len(y)
    for i in range(samples):
        if (digit == y[i]):
            if (y[i] == y_hat[i]):
                mat[0][0] += 1
            else:
                mat[1][0] += 1
        else:
            if (y[i] == y_hat[i]):
                mat[1][1] += 1
            else:
                mat[0][1] += 1

    for i in range(0,2):
        for j in range(0,2):
            mat[i][j] = mat[i][j] / samples

    return mat

def matthew_coeff(mat):
    num = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    deno = (mat[0][0] + mat[0][1]) * (mat[0][0] + mat[1][0]) * (mat[1][1] + mat[0][1]) * (mat[1][0] + mat[1][1])
    deno = math.sqrt(deno)
    return num/deno

def for_each_digit(y_hat,y):
    all_coeff = dict()
    for i in range(0,10):
        all_coeff[i] = 0

    min_c = 0
    least_conf = 0
    for i in range(0,10):
        temp = visualise_confusion(y_hat,y,i)
        coeff = matthew_coeff(temp)
        all_coeff[i] = coeff
        if (min_c > coeff):
            min_c = coeff
            least_conf = i
    
    max_c_1 = 0
    max_conf_1 = 0
    for i in range(0,10):
        if (max_conf_1 < all_coeff[i]):
            max_conf_1 = all_coeff[i]
            max_c_1 = i

    all_coeff[max_c_1] = -1
    max_c_2 = 0
    max_conf_2 = 0
    for i in range(0,10):
        if (max_conf_2 < all_coeff[i]):
            max_conf_2 = all_coeff[i]
            max_c_2 = i
    
    return (min_c, max_c_1, max_c_2)

print(for_each_digit(y_hat,y_test))