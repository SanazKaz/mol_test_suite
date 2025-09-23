import pytest
from src.metrics_2d import calculate_qed

# --- Test Case 1: A valid, well-known molecule ---
def test_calculate_qed_for_benzene():
    """
    Tests the QED calculation for a simple, valid SMILES string (Benzene).
    """
    # 1. Arrange: Define the input and the expected output.
    smiles = "c1ccccc1"  # Benzene
    expected_qed = 0.4462 # This is the known QED for Benzene

    # 2. Act: Call the function we are testing.
    actual_qed = calculate_qed(smiles)

    # 3. Assert: Check that the actual result matches the expected result.
    # We use pytest.approx because floating-point numbers can have tiny precision errors.
    assert actual_qed == pytest.approx(expected_qed, abs=1e-4)


# --- Test Case 2: An invalid molecule ---
def test_calculate_qed_for_invalid_smiles():
    """
    Tests that the function correctly handles an invalid SMILES string.
    """
    # 1. Arrange
    smiles = "this is not a valid smiles"
    expected_result = -1.0

    # 2. Act
    actual_result = calculate_qed(smiles)

    # 3. Assert
    assert actual_result == expected_result


# --- Test Case 3: A more complex, drug-like molecule ---
def test_calculate_qed_for_caffeine():
    """
    Tests the QED calculation for a more complex molecule (Caffeine).
    """
    # Arrange
    smiles = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C" # Caffeine
    expected_qed = 0.6225

    # Act
    actual_qed = calculate_qed(smiles)

    # Assert
    assert actual_qed == pytest.approx(expected_qed, abs=1e-4)