import sys
import copy
import operator
import numpy as np
# import subprocess

base_table = []
table_user = []

user_average = []
item_average = []
item_norm = []

cache_user = []
cache_item = []
test_cnt = None
    
def readData(file_name):
    fin = open(file_name,'r')
    data = {} #dict 
    while True:
        tmp = fin.readline() #user item rating time
        if not tmp:
            break
        tmp = list(tmp.strip('\n').split('\t'))
        tmp.pop() # delete time
        for i in range(len(tmp)):
            tmp[i] = int(tmp[i])
        if not tmp[0] in data:
            data[tmp[0]] = [tmp[1:]]
        elif tmp[0] in data:
            data[tmp[0]].append(tmp[1:])
    fin.close()
    # user: [item,rating],[item,rating],....
    return data

def writeData(file_name,data):
    fout = open(file_name,'w+')
    fout.write('\n'.join(data))
    fout.close()


# cosine-based similarity
def simItem(i,j):
    global base_table, item_norm, cache_item
    col_i = base_table[:,i]
    col_j = base_table[:,j]
    norm2_i = item_norm[i]
    norm2_j = item_norm[j]
    if norm2_i == 0  or norm2_j == 0:
        ret = 0
    else:
        ret = np.dot(col_i,col_j)/(norm2_i*norm2_j)
    cache_item[i][j] = ret
    cache_item[j][i] = ret
    return ret


def itemBasedPrediction(target_user,target_item):
    global base_table, cache_item, user_average
    similarity = {} # target_item 과 의 similarity 전부다 구하기
    for i in range(1,base_table.shape[1]): # item 전체에 대해서
        if i == target_item:
            continue
        if cache_item[target_user][i] != -1:
            similarity[i] = cache_item[target_user][i]
        else:
            similarity[i] = simItem(target_user,i)
    top = 0; bottom = 0; tmp = {}
    for i,rate in enumerate(base_table[target_user]):
        if rate != 0 and similarity[i] > 0:
            tmp[i] = similarity[i]
    tmp = sorted(tmp.items(),key=operator.itemgetter(1),reverse=True)
    for i,s in tmp:
            bottom += s
            top += base_table[target_user][i]*s
    if top==0:
        return user_average[target_user]
    else:          
        return top/bottom

# 유저 평균 보정해줌 
def simUser(i,j):
    global table_user, cache_user
    user_i = table_user[i]
    user_j = table_user[j]
    ret = np.dot(user_i,user_j)/(np.sqrt(np.dot(user_i,user_i))*np.sqrt(np.dot(user_j,user_j)))
    cache_user[i][j] = ret  
    cache_user[j][i] = ret  
    return ret

    
def userBasedPrediction(target_user,target_item):
    global base_table, cache_user, user_average
    similarity = {}
    for i in range(1,base_table.shape[0]): #user 전체에 대해서
        if base_table[i][target_item] == 0 or i == target_user: 
            continue #본인이거나 rating이 안된 경우 안구함
        if cache_user[target_user][i] != -1:
            similarity[i] = cache_user[target_user][i]
        else:
            similarity[i] = simUser(target_user,i)
    top = 0; bottom = 0; tmp = {}
    for i in range(1,len(base_table)):        
        if i == target_user: # 본인이면 패스
            continue
        if i in similarity: # similarity 구한 경우만
            if similarity[i] > 0:
                tmp[i] = similarity[i]
    tmp = sorted(tmp.items(),key=operator.itemgetter(1),reverse=True)
    tmp = tmp[:50]
    for i,s in tmp:
            bottom += s
            top += base_table[i][target_item]*s
    if top==0:
        return user_average[target_user]
    else:          
        return top/bottom

    

def prediction(test_data):
    global test_cnt
    cnt = 0
    output_data = []
    for key in test_data.keys():
        for rating in test_data[key]:
            user = key
            item = rating[0]
            predict = 0.5*userBasedPrediction(user,item) + 0.5*itemBasedPrediction(user,item)
            if predict == 0:
                predict = 1
            tmp = "{}\t{}\t{}".format(user,item,predict)
            output_data.append(tmp)
            cnt += 1
            if cnt%10000 == 0:
                print("{}/{}".format(cnt,test_cnt))
    return output_data


def preprocess(base_data, test_data):
    global user_average, item_norm
    global table_user, base_table
    global cache_item, cache_user
    global test_cnt

    # total test count 
    test_cnt = 0
    for key in test_data.keys():
        test_cnt += len(test_data[key])

    # item_size and user_size
    item_size = 0; user_size = 0
    for key in base_data.keys():
        for item in base_data[key]:
            if item[0] > item_size:
                item_size = item[0]
        if key > user_size:
            user_size = key
    for key in test_data.keys():
        for item in test_data[key]:
            if item[0] > item_size:
                item_size = item[0]
        if key > user_size:
            user_size = key

    # make table based on base_data
    base_table = np.zeros((user_size+1,item_size+1),dtype=float)
    for key in base_data.keys():
        for rating in base_data[key]:
            user = key
            item = rating[0]
            rate = rating[1]
            base_table[user][item] = rate
    
    # average rating of each user 
    for i,user in enumerate(base_table):
        if i == 0:
            user_average.append(0)
        else:
            user_average.append(np.sum(user)/np.count_nonzero(user))

    # norm of each item
    for i,item in enumerate(base_table.T):
        if i == 0:
            item_norm.append(0)
        else:
            item_norm.append(np.sqrt(np.dot(item,item)))

    # user의 평균을 뺴서 보정해줌 
    table_user = copy.deepcopy(base_table)
    for i in range(1,len(table_user)):
        for index in np.nonzero(table_user[i])[0]:
            table_user[i][index] -= user_average[i]
    
    # 속도 향상을 위해 cache table 
    cache_item = np.full((item_size+1,item_size+1),-1,dtype='float64')
    cache_user = np.full((user_size+1,user_size+1),-1,dtype='float64')


def main():
    base_file = sys.argv[1]
    test_file = sys.argv[2]

    if base_file.split('.')[0] != test_file.split('.')[0]:
        print("base test file do not match")
        exit()

    name = base_file.split('.')[0]
    base_data = readData(base_file)
    test_data = readData(test_file)

    preprocess(base_data,test_data)
    output_data = prediction(test_data)    
    writeData("{}.base_prediction.txt".format(name),output_data)

    # out = subprocess.check_output("PA4.exe {}".format(name),shell=True, encoding='utf-8')
    # result = float(out.split('\n')[-2].split()[-1])
    # print(result)

    
if __name__ == "__main__":
    main()
    