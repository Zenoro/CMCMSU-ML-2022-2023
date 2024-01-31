from task7 import find_shortest
import pytest


@pytest.mark.parametrize(
    "arg,res",
    [
        [";123assdcdcef092101,3131313akdmkmedkfmwekfwe", 9],
        ['tr9230847;;;;1;;;++++_______abbbbbbc', 2],
        ["askdhrfef8wej9013d,kdj;12oid3fjvn23", 1],
        ['9230847;;;;1;;;++++_______a', 1],
        ['жж1ciwoeiworiworworow', 18],
        ["aslkdjfhkssdf", 13],
        ["ad2aaaaa,,bsc", 2],
        ['12i330232l', 1],
        ["111", 0],
        ["+_*", 0],
        ["", 0]
    ]
)
def test_find_shortest(arg, res):
    assert find_shortest(arg) == res
