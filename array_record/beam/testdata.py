"""Simple test data wrapper.

Separated to keep Tensorflow and Beam dependencies away from test data
"""

# Hardcoded multirecord dataset in dict format for testing and demo.
data = [
    {
        'Age': 29,
        'Movie': ['The Shawshank Redemption', 'Fight Club'],
        'Movie Ratings': [9.0, 9.7],
        'Suggestion': 'Inception',
        'Suggestion Purchased': 1.0,
        'Purchase Price': 9.99
    },
    {
        'Age': 39,
        'Movie': ['The Prestige', 'The Big Lebowski', 'The Fall'],
        'Movie Ratings': [9.5, 8.5, 8.5],
        'Suggestion': 'Interstellar',
        'Suggestion Purchased': 1.0,
        'Purchase Price': 14.99
    },
    {
        'Age': 19,
        'Movie': ['Barbie', 'The Batman', 'Boss Baby', 'Oppenheimer'],
        'Movie Ratings': [9.6, 8.2, 10.0, 4.2],
        'Suggestion': 'Secret Life of Pets',
        'Suggestion Purchased': 0.0,
        'Purchase Price': 25.99
    },
    {
        'Age': 35,
        'Movie': ['The Mothman Prophecies', 'Sinister'],
        'Movie Ratings': [8.3, 9.0],
        'Suggestion': 'Hereditary',
        'Suggestion Purchased': 1.0,
        'Purchase Price': 12.99
    }
]
