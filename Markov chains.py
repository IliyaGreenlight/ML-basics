import numpy
import matplotlib.pyplot as plt

possibleSigns = ['R', 'P', 'S']
Initial_probability = [0.23, 0.44, 0.33]
sumOfPointsEarned = 0
xCoordinates = []
yCoordinates = []
learningStep = 0.0002
victoryPointGoal = 200

Transition_matrix = [
                            #       Rock | Paper | Scissors
    [0.5, 0.43, 0.07],      # Rock
    [0.24, 0.36, 0.40],     # Paper
    [0.1, 0.6, 0.3]         # Scissors
]

Emission_matrix = [
                            #         0  |   1   |    -1
    [0.28, 0.42, 0.3],      # Rock
    [0.15, 0.23, 0.62],     # Paper
    [0.4, 0.5, 0.1]         # Scissors
]


# Init
state = numpy.random.choice(possibleSigns, replace=True, p=Initial_probability)
for x in range(1, 10000):
    if (abs(sumOfPointsEarned) >= victoryPointGoal):
        break
    response = ''
    # Generating initial response
    if state == 'R':
        response = numpy.random.choice(possibleSigns, replace=True, p=Transition_matrix[0])
    elif state == 'P':
        response = numpy.random.choice(possibleSigns, replace=True, p=Transition_matrix[1])
    else:
        response = numpy.random.choice(possibleSigns, replace=True, p=Transition_matrix[2])

    # Checking win or lose
    if state == 'R':
        state = numpy.random.choice(possibleSigns, replace=True, p=Emission_matrix[0])
        if response == 'R':
            result = 0
        elif response == 'P':
            result = 1
            if (Transition_matrix[0][1] + learningStep) < 1 and (Transition_matrix[0][0] - learningStep / 2) > 0 \
                    and (Transition_matrix[0][2] - learningStep / 2) > 0:
                Transition_matrix[0][1] += learningStep
                Transition_matrix[0][0] -= learningStep / 2
                Transition_matrix[0][2] -= learningStep / 2
        else:
            result = -1

    elif state == 'P':
        state = numpy.random.choice(possibleSigns, replace=True, p=Emission_matrix[1])
        if response == 'R':
            result = -1
        elif response == 'P':
            result = 0
        else:
            result = 1

            if (Transition_matrix[1][2] + learningStep) < 1 and (Transition_matrix[1][0] - learningStep / 2) > 0 \
                    and (Transition_matrix[1][1] - learningStep / 2) > 0:
                Transition_matrix[1][2] += learningStep
                Transition_matrix[1][0] -= learningStep / 2
                Transition_matrix[1][1] -= learningStep / 2

    else:
        state = numpy.random.choice(possibleSigns, replace=True, p=Emission_matrix[2])
        if response == 'R':
            result = 1
            if (Transition_matrix[2][0] + learningStep) < 1 and (Transition_matrix[2][1] - learningStep / 2) > 0 \
                    and (Transition_matrix[2][2] - learningStep / 2) > 0:
                Transition_matrix[2][0] += learningStep
                Transition_matrix[2][1] -= learningStep / 2
                Transition_matrix[2][2] -= learningStep / 2
        elif response == 'P':
            result = -1
        else:
            result = 0


    sumOfPointsEarned = sumOfPointsEarned + result
    xCoordinates.append(x)
    yCoordinates.append(sumOfPointsEarned)

# Plotting
plt.plot(xCoordinates, yCoordinates)
plt.xlabel("Steps")
plt.ylabel("Victory points")
plt.show()
