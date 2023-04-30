def get_data():
    return [[-1,1],[1,-1],[-1,-1],[1,1]]

def get_data_expected(operation):
    if(operation == 'AND'):
        result = [-1,-1,-1,1] 
    else:
        result = [1,1,-1,-1] 
    return result
