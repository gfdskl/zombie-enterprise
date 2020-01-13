from sklearn.metrics import f1_score,accuracy_score,recall_score

def evalModel(y,pred_y):
    f1 = f1_score(y,pred_y)
    accu = accuracy_score(y,pred_y)
    recall = recall_score(y,pred_y)

    print ("f1_score:{}\naccuracy_score:{}\nrecall_score:{}".format(f1,accu,recall))





