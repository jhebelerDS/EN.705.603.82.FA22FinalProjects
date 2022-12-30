'''This script will prompt the user for a factored SINGLE-VARIATE polynomial sequence.
It will use the trained model to predict the distributed form of the polynomial.'''

from model import load_model, predict_single_sentence

def prompt_user(model):
    '''params: model - the trained model
    returns: Console output of the prediction, or False if the input is invalid'''
    # Prompt the user for input
    print("Please enter a factored single-variate polynomial sequence:")
    sentence = input()

    # Make a prediction
    try:
        prediction = predict_single_sentence(model, sentence)
    except:
        return False

    # Print the prediction
    print("Prediction:", prediction)
    print()

def main():
    print("Loading model...")

    # Load the model
    try:
        model = load_model()
        print('The model is loaded and ready to use.')
        print()
    except Exception:
        print("Model not found. Please train the model first.")
        return

    with open('vocab.txt', 'r') as f:
        vocab = f.read().splitlines()

    print('Please use the following vocabulary:')
    print(vocab)
    print('Each character is considered a word in the vocabulary.')
    print('Use ** to represent exponentiation, NOT ^')
    print()
    print('Examples of valid input:')
    print("Input: x*(x+2)")
    print("Target: x**2+2*x")
    print()
    print("Input: -4*s*(-4*s-27)")
    print("Target: 16*s**2+108*s")
    print()

    while True:
        ans = prompt_user(model)

        if ans is False:
            print("Invalid input. Please try again.")
            continue

        print('Would you like to try another sequence? (y/n)')
        response = input()
        print()
        if response.lower() != 'y':
            print('Exiting program...')
            break


if __name__ == "__main__":
    print('System Project For Creating AI Enabled Systems')
    print('Class: EN.705.603')
    print('Programmer: Chris Vaisnor')
    print('This program will prompt the user for a factored SINGLE-VARIATE polynomial sequence.')
    print('It will use the trained model to predict the distributed form of the polynomial.')
    print('--------------------------------------------------------------------------------')
    main()
