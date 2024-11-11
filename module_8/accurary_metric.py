from sklearn.metrics import accuracy_score

def simple_binary():
    # Simple binary classification sample
    prediction = [1,1,1]
    actual_result = [1,0,1]

    correct = 0
    for index in range(len(prediction)):
        if prediction[index] == actual_result[index]:
            correct += 1
    print(f'For loop accurary = {(correct/len(prediction))*100:.2f}')
    print(f'For Sklearn accurary = {(accuracy_score(actual_result,prediction))*100:.2f}')




def main():
    simple_binary()


if __name__ == '__main__':
    main()