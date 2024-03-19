import sorobn as hh
import pandas as pd
import random

#part a
#bayesian network initialized here, makes markov chains
bn = hh.BayesNet(("C", ["S", "R"]), ("S", "W"), ("R", "W"))
#set the probabilites here
bn.P["C"] = pd.Series({True: 0.5, False:0.5})
bn.P["S"] = pd.Series({(True, True): 0.1, (True, False): 0.9, (False, True): 0.5, (False, False): 0.5})
bn.P["R"] = pd.Series({(True, True): 0.8, (True, False): 0.2, (False, True): 0.2, (False, False): 0.8})
bn.P["W"] = pd.Series({ (True, True, True): 0.99, (True, True, False): 0.01, (True, False, True): 0.9, (True, False, False): 0.1,
                       (False, True, True): 0.95, (False, True, False): 0.05,(False, False, True): 0.05, (False, False, False): 0.95})

#prepare for inference
bn.prepare()
probC = bn.query("C", event={"S": False, "W": True})

#part b
# c given not s and r
probC_given_r = bn.query("C", event={"S": False, "R": True})
# c given not s and not r
probC_given_notr = bn.query("C", event={"S": False, "R": False})
# r given c, not s, and w
probR_given_c = bn.query("R", event={"C": True, "S": False, "W": True})
# r given not c, not s, and w
probR_given_notc= bn.query("R", event={"C": False, "S": False, "W": True})

#part c
# Define rows and columns
rows = 4
columns = 4
# Create a matrix filled with zeros
Q = [[0 for _ in range(rows)] for _ in range(columns)]

#probability of (c,r) -> (c,r) S1
Q[0][0] = 0.5 * probC_given_r[True] + 0.5 * probR_given_c[True]
#probability of (c,r) -> (c,-r) S2
Q[0][1] = 0.5 * probR_given_c[False]
#probability of (c,r) -> (-c,r) S3
Q[0][2] = 0.5 * probC_given_r[False]
#probability of (c,r) -> (-c,-r) S4
Q[0][3] = 0

#probability of (c,-r) -> (c,r)
Q[1][0] = 0.5 * probR_given_c[True] 
# #probability of (c,-r) -> (c,-r)
Q[1][1] = 0.5 * probC_given_notr[True] +  0.5 * probR_given_c[False]
# #probability of (c,-r) -> (-c,r)
Q[1][2] = 0
# #probability of (c,-r) -> (-c,-r)
Q[1][3] = 0.5 * probC_given_notr[False]

#probability of (-c,r) -> (c,r)
Q[2][0] = 0.5 * probC_given_r[True]
#probability of (-c,r) -> (c,-r)
Q[2][1] = 0
#probability of (-c,r) -> (-c,r)
Q[2][2] = 0.5 * probC_given_r[False] + 0.5 * probR_given_notc[True]
#probability of (-c,r) -> (-c,-r)
Q[2][3] = 0.5 * probR_given_notc[False]

#probability of (-c,-r) -> (c,r)
Q[3][0] = 0
#probability of (-c,-r) -> (c,-r)
Q[3][1] = 0.5 * probC_given_notr[True]
#probability of (-c,-r) -> (-c,r)
Q[3][2] = 0.5 * probR_given_notc[True]
#probability of (-c,-r) -> (-c,-r)
Q[3][3] = 0.5 * probC_given_notr[False] + 0.5 * probR_given_notc[False]

#part d
nArr = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
exProb = probC[True] 

#create temp Q
def tempQ(Q):
    tempQ = []
    #loop through a row of Q
    for i in range(rows):
        #make an empty element
        tempQ.append([])
        #add row of Q to tempQ
        tempQ[i].extend(Q[i])
    return tempQ

#return arr of estimated probabilities   
def getProb():
    temp = tempQ(Q)
    estProbArr = []
    #get n times in nArr
    for n in nArr:
        count = 0
        # Start at state 1
        state = 0
        #run n times
        for _ in range(n):
            # if state is at S1 or S2(where c is true)
            if state == 0 or state == 1:
                #counter updates
                count += 1
            #transitions state based on the probabilities in the transition matrix
            #choice returns a random number in an array based on the weights of that row 
            state = random.choices(range(rows), weights = temp[state])[0]
        estProb = count / n
        # append total estProb at the end of for loop
        estProbArr.append(estProb)
    return estProbArr

#get array of error values
def getError(probArr):
    errorArr = []
    #calculate errors based on n
    for estProb in probArr:
        error = abs(estProb - exProb) / exProb * 100
        errorArr.append(error)
    return errorArr

probArr = getProb()
errorArr = getError(probArr)

#Print probabilities of events being true and false
print("Part A. The sampling probabilites")
print(f"P(C|-s, r) = <{probC_given_r[True]:.4f}, {probC_given_r[False]:.4f}>")
print(f"P(C|-s, -r) = <{probC_given_notr[True]:.4f}, {probC_given_notr[False]:.4f}>")
print(f"P(R|c, -s, w) = <{probR_given_c[True]:.4f}, {probR_given_c[False]:.4f}>")
print(f"P(R|-c, -s, w) = <{probR_given_notc[True]:.4f}, {probR_given_notc[False]:.4f}>")

stateArr = ["S1", "S2", "S3", "S4"]
headerStr = "   "
print("\nPart B. The transition probability matrix")
for i in range(len(stateArr)):
    #add elements of stateArr for header
    headerStr += stateArr[i] + "     "
print(headerStr)

for i in range(rows):
    rowStr = ""
    #add stateArr at the beginning
    rowStr += stateArr[i]
    for j in range(columns):
        #add every element in Q's row to rowStr
        rowStr += " " + f"{Q[i][j]:.4f}"
    print(rowStr)


print("\nPart C. The probability for the query P(C|-s, w)")
#print out the probability of C
print(f"Exact Probability: <{probC[True]:.4f}, {probC[False]:.4f}>")
counter = 0
for n in range(3, 7):
    #print estimated probabilities and errors from arr
    print(f"n = 10 ^ {n}: <{probArr[counter]:.4f}, {(1 - probArr[counter]):.4f}>, error = {errorArr[counter]:.2f} %")  
    counter += 1

