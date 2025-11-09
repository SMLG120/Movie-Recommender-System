# Unit Testing

## Fixtures

3 [fixtures](https://docs.pytest.org/en/stable/explanation/fixtures.html) are defined to provide consistency for the unit tests: 

- my_user: user_id, age, occupation, gender
- my_movies_dataframe: id, title, original_language, release_date, runtime, popularity, vote_average, vote_count, genres
- my_model: mock model instantiated by [MagicMock](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.MagicMock)

## Patch Decorator
The patch decorator functions to mock file access, databse acsess, and function calls so that methods can be tested in isolation of other elements of a system. 

[Patch Documentation](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.patch)

## Unit Tests

### test_init
This test asserts that RecommenderEngine instantiates a model that contains a dataframe called movies and this dataframe movies contains a column "title".

#### Parameters
- mock_read_csv: mock call to pd.read_csv called in RecommenderEngine.__init__; return value set to fixture dataframe of movie information
- mock_joblib_load: mock call to joblib.load(model_path) in RecommenderEngine.__init__; return value set to fixture of a mock model
- my_movies_dataframe: fixture
- my_model: fixture

### test_get_user_info_works
This test asserts that when a user exists and get_user_info is called on their id, the system will retrieve the correct user.

#### Parameters
- mock_joblib_load: mock model loaded
- mock_read_csv: mock call to pd.read_csv which returns a dataframe of parsed movie information; return value set to mock fixture my_movies_dataframe
- mock_requests_get: mimics fetching user info from the API
- my_user: fixture
- my_model: fixture

### test_get_user_info_fails
This test asserts that when a user does not exist and get_user_info is called on their id, the system will set their age to -1 and gender to U.

#### Parameters
- mock_joblib_load: mock model loaded
- mock_read_csv: mock call to pd.read_csv which returns a dataframe of parsed movie information; return value set to mock fixture my_movies_dataframe
- mock_requests_get: mimics fetching user info from the API
- my_model: fixture
- my_movies_dataframe: fixture

## To Be Added

### test_build_inference_dataframe

### test_recommend

## Resources

- [pytest documentation](https://docs.pytest.org/en/stable/contents.html)
- [fixtures](https://docs.pytest.org/en/stable/explanation/fixtures.html)
- [MagicMock](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.MagicMock)
- [Patch Decorator](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.patch)

