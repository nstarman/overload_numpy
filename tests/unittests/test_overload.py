"""Unit tests for :mod:`overload_numpy.overload`."""

##############################################################################
# IMPORTS

# THIRDPARTY
import pytest

# LOCAL
from overload_numpy.overload import NumPyOverloader

##############################################################################
# TESTS
##############################################################################


class Test_NumPyOverloader:
    """Test :class:`overload_numpy.NumPyOverloader`."""

    @pytest.fixture
    def overloader(self) -> NumPyOverloader:
        overloader = NumPyOverloader()

        # TODO! mock up items

        return overloader

    # ===============================================================
    # Method unit tests

    def test_init(self, overloader):
        """If the fixture passes, so does this."""

    @pytest.mark.skip(reason="TODO")
    def test___getitem__(self, overloader):
        pass

    @pytest.mark.skip(reason="TODO")
    def test___contains__(self, overloader):
        pass

    @pytest.mark.skip(reason="TODO")
    def test___iter__(self, overloader):
        pass

    @pytest.mark.skip(reason="TODO")
    def test___len__(self, overloader):
        pass

    @pytest.mark.skip(reason="TODO")
    def test_keys(self, overloader):
        pass

    @pytest.mark.skip(reason="TODO")
    def test_values(self, overloader):
        pass

    @pytest.mark.skip(reason="TODO")
    def test__parse_types(self, overloader):
        pass

    @pytest.mark.skip(reason="TODO")
    def test_implements(self, overloader):
        pass

    @pytest.mark.skip(reason="TODO")
    def test_assists(self, overloader):
        pass

    # ===============================================================
    # Usage tests

    @pytest.mark.incompatible_with_mypyc
    def test_serialization(self, overloader):
        pass
