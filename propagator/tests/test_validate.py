import nose.tools as nt

from propagator import validate


def test_flow_direction_good():
    nt.assert_equal(
        validate.flow_direction("uPSTReam"),
        "upstream"
    )

    nt.assert_equal(
        validate.flow_direction("downSTReam"),
        "downstream"
    )


@nt.raises(ValueError)
def test_flow_direction_bad():
    validate.flow_direction("sideways")


class Test_non_empty_list(object):
    def test_baseline(self):
        x = [1, 2, 3]
        nt.assert_list_equal(validate.non_empty_list(x), x)

    def test_scalar(self):
        x = 1
        nt.assert_list_equal(validate.non_empty_list(x), [x])

    @nt.raises(ValueError)
    def test_None_raises(self):
        validate.non_empty_list(None)

    def test_None_creates(self):
        nt.assert_list_equal(validate.non_empty_list(None, on_fail='create'), [])

    @nt.raises(ValueError)
    def test_empty_list_raises(self):
        validate.non_empty_list([])

    def test_empty_creates(self):
        nt.assert_list_equal(validate.non_empty_list([], on_fail='create'), [])


def test_value_column_stats():
    value_cols = [
        ('test1', 'mean'),
        'test2',
        ('test3', 'median'),
        'test4',
        ('test5',),
    ]

    expected = [
        ('test1', 'mean'),
        ('test2', 'mean'),
        ('test3', 'median'),
        ('test4', 'mean'),
        ('test5', 'mean'),
    ]

    result = validate.value_column_stats(value_cols, 'mean')
    nt.assert_list_equal(result, expected)
