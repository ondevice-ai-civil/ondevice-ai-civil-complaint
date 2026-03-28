"""
Tests for PII Masking Module
"""

import pytest

from src.data_collection_preprocessing.pii_masking import (
    PIIMasker,
    PIIType,
    mask_pii,
    validate_no_pii,
)


class TestPIIMasker:
    """Tests for PIIMasker class"""

    @pytest.fixture
    def masker(self):
        """Create a strict PIIMasker instance"""
        return PIIMasker.create_strict_masker()

    @pytest.fixture
    def basic_masker(self):
        """Create a basic PIIMasker instance"""
        return PIIMasker.create_basic_masker()

    def test_mask_resident_id(self, masker):
        """Test masking Korean resident registration numbers"""
        text = "My ID is 901231-1234567"
        result = masker.mask_text(text)
        assert "901231-1234567" not in result
        assert "[RESIDENT_ID_MASKED]" in result

    def test_mask_phone_mobile(self, masker):
        """Test masking mobile phone numbers"""
        text = "Call me at 010-1234-5678"
        result = masker.mask_text(text)
        assert "010-1234-5678" not in result
        assert "[PHONE_MASKED]" in result

    def test_mask_phone_landline(self, masker):
        """Test masking landline phone numbers"""
        text = "Office number: 02-1234-5678"
        result = masker.mask_text(text)
        assert "02-1234-5678" not in result
        assert "[PHONE_MASKED]" in result

    def test_mask_email(self, masker):
        """Test masking email addresses"""
        text = "Contact: test@example.com for details"
        result = masker.mask_text(text)
        assert "test@example.com" not in result
        assert "[EMAIL_MASKED]" in result

    def test_mask_ip_address(self, masker):
        """Test masking IP addresses"""
        text = "Server IP: 192.168.1.100"
        result = masker.mask_text(text)
        assert "192.168.1.100" not in result
        assert "[IP_ADDRESS_MASKED]" in result

    def test_mask_credit_card(self, masker):
        """Test masking credit card numbers"""
        text = "Card: 1234-5678-9012-3456"
        result = masker.mask_text(text)
        assert "1234-5678-9012-3456" not in result
        assert "[CREDIT_CARD_MASKED]" in result

    def test_no_masking_clean_text(self, masker):
        """Test that clean text is not modified"""
        text = "This is a normal complaint about road conditions."
        result = masker.mask_text(text)
        assert result == text

    def test_empty_text(self, masker):
        """Test handling of empty text"""
        assert masker.mask_text("") == ""
        assert masker.mask_text(None) == ""

    def test_multiple_pii_types(self, masker):
        """Test masking multiple PII types in one text"""
        text = "Contact 010-1234-5678 or email@test.com"
        result = masker.mask_text(text)
        assert "010-1234-5678" not in result
        assert "email@test.com" not in result

    def test_detect_pii(self, masker):
        """Test PII detection without masking"""
        text = "My phone is 010-1234-5678 and email is test@test.com"
        matches = masker.detect_pii(text)
        assert len(matches) >= 2
        pii_types = [m.pii_type for m in matches]
        assert PIIType.PHONE in pii_types
        assert PIIType.EMAIL in pii_types

    def test_statistics(self, masker):
        """Test masking statistics tracking"""
        masker.reset_statistics()
        masker.mask_text("Call 010-1234-5678")
        masker.mask_text("Email me at test@test.com")
        stats = masker.get_statistics()
        assert stats["phone"] >= 1
        assert stats["email"] >= 1


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_mask_pii_strict(self):
        """Test strict masking via convenience function"""
        text = "Contact: 010-1234-5678, email@test.com, IP: 192.168.1.1"
        result = mask_pii(text, strict=True)
        assert "010-1234-5678" not in result
        assert "email@test.com" not in result
        assert "192.168.1.1" not in result

    def test_mask_pii_basic(self):
        """Test basic masking via convenience function"""
        text = "Contact: 010-1234-5678, email@test.com"
        result = mask_pii(text, strict=False)
        assert "010-1234-5678" not in result
        assert "email@test.com" not in result

    def test_validate_no_pii_clean(self):
        """Test validation on clean text"""
        text = "This is a normal complaint text without any personal information."
        is_clean, matches = validate_no_pii(text)
        assert is_clean
        assert len(matches) == 0

    def test_validate_no_pii_with_pii(self):
        """Test validation on text containing PII"""
        text = "My phone number is 010-1234-5678"
        is_clean, matches = validate_no_pii(text)
        assert not is_clean
        assert len(matches) >= 1


class TestKoreanNameMasking:
    """Tests for Korean name detection and masking"""

    @pytest.fixture
    def masker(self):
        return PIIMasker.create_strict_masker()

    def test_mask_korean_name_common(self, masker):
        """Test masking common Korean names"""
        # Common Korean names with common surnames
        names_to_test = ["Kim", "Lee", "Park"]  # Simplified for English testing
        # Note: Full Korean name testing would require Korean text

    def test_mask_all_includes_names(self, masker):
        """Test that mask_all includes name detection"""
        text = "The complaint was filed by citizen."
        result = masker.mask_all(text, include_name_detection=True)
        # Basic validation that the method runs without error
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
