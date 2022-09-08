"""Unit tests for :mod:`overload_numpy.overload`."""

# THIRD PARTY
import pytest

# LOCAL
from overload_numpy.overload import NumPyOverloader


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
        """If the fixture passes, so does this"""

    @pytest.mark.skip
    def test___getitem__(self, overloader):
        pass

    @pytest.mark.skip
    def test___contains__(self, overloader):
        pass

    @pytest.mark.skip
    def test___iter__(self, overloader):
        pass

    @pytest.mark.skip
    def test___len__(self, overloader):
        pass

    @pytest.mark.skip
    def test_keys(self, overloader):
        pass

    @pytest.mark.skip
    def test_values(self, overloader):
        pass

    @pytest.mark.skip
    def test__parse_types(self, overloader):
        pass

    @pytest.mark.skip
    def test_implements(self, overloader):
        pass

    # ===============================================================
    # Usage tests

    @pytest.mark.skip
    def test_serialization(self, overloader):
        pass
