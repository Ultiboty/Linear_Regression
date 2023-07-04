import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def gradient_decent(m_now, b_now, points, L_rate):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L_rate
    b = b_now - b_gradient * L_rate
    return m, b


def main():
    # Importing the dataset
    dataset = pd.read_csv('Salary_Data.csv')

    # vars
    m = 0
    b = 0
    L_rate = 0.01
    epochs = 300

    # train the modle
    for i in range(epochs):
        if i % 50 == 0:
            print(f'Epochs: {i}')

        m, b = gradient_decent(m, b, dataset, L_rate)
    print('m =', m, ' b =', b)

    # make the linear line
    data = [(m * round(x, 2)) + b for x in np.arange(1, 10, 0.3)]

    # Visualising the Test set results
    plt.scatter(dataset.YearsExperience, dataset.Salary, color='red')
    plt.plot([x for x in np.arange(1, 10, 0.3)], data, color='blue')
    plt.title('Salary vs Experience (Test set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.figtext(.7, .2, "Red dot = data\nLine = Prediction")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()