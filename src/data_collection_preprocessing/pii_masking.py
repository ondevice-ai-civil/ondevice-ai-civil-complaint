"""
PII (Personal Identifiable Information) Masking Module

Detects and masks personal information in civil complaint text data.
Supports Korean-specific PII patterns including:
- Korean resident registration numbers
- Phone numbers (Korean formats)
- Email addresses
- Korean names (basic pattern)
- Physical addresses
- Bank account numbers
- Credit card numbers
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected and masked"""

    RESIDENT_ID = "resident_id"  # Korean resident registration number
    PHONE = "phone"  # Phone numbers
    EMAIL = "email"  # Email addresses
    NAME = "name"  # Person names
    ADDRESS = "address"  # Physical addresses
    BANK_ACCOUNT = "bank_account"  # Bank account numbers
    CREDIT_CARD = "credit_card"  # Credit card numbers
    PASSPORT = "passport"  # Passport numbers
    DRIVER_LICENSE = "driver_license"  # Driver's license numbers
    VEHICLE_PLATE = "vehicle_plate"  # Vehicle license plates
    IP_ADDRESS = "ip_address"  # IP addresses


@dataclass
class PIIMatch:
    """Represents a detected PII match"""

    pii_type: PIIType
    original_text: str
    start_pos: int
    end_pos: int
    masked_text: str


@dataclass
class PIIPattern:
    """PII detection pattern configuration"""

    pii_type: PIIType
    pattern: str
    mask_template: str
    description: str
    priority: int = 0  # Higher priority patterns are checked first


# Korean PII patterns
PII_PATTERNS: List[PIIPattern] = [
    # Korean Resident Registration Number (주민등록번호)
    # Format: YYMMDD-GNNNNNN (13 digits with hyphen)
    PIIPattern(
        pii_type=PIIType.RESIDENT_ID,
        pattern=r"\b(\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))\s*[-]\s*([1-4]\d{6})\b",
        mask_template="[RESIDENT_ID_MASKED]",
        description="Korean resident registration number",
        priority=10,
    ),
    # Korean Phone Numbers
    # Mobile: 010-XXXX-XXXX, 010XXXXXXXX
    PIIPattern(
        pii_type=PIIType.PHONE,
        pattern=r"\b(01[016789])[-.\s]?(\d{3,4})[-.\s]?(\d{4})\b",
        mask_template="[PHONE_MASKED]",
        description="Korean mobile phone number",
        priority=8,
    ),
    # Landline: 02-XXXX-XXXX, 031-XXX-XXXX
    PIIPattern(
        pii_type=PIIType.PHONE,
        pattern=r"\b(0[2-6][0-5]?)[-.\s]?(\d{3,4})[-.\s]?(\d{4})\b",
        mask_template="[PHONE_MASKED]",
        description="Korean landline phone number",
        priority=7,
    ),
    # Email addresses
    PIIPattern(
        pii_type=PIIType.EMAIL,
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        mask_template="[EMAIL_MASKED]",
        description="Email address",
        priority=6,
    ),
    # Korean addresses (simplified pattern)
    # Matches patterns like "서울시 강남구 xxx동 xxx번지"
    PIIPattern(
        pii_type=PIIType.ADDRESS,
        pattern=r"([가-힣]{2,4}(?:시|도))\s*([가-힣]{2,4}(?:시|군|구))\s*([가-힣]{2,10}(?:읍|면|동|로|길))\s*(\d{1,5}(?:번지|번)?(?:-\d{1,5})?)",
        mask_template="[ADDRESS_MASKED]",
        description="Korean physical address",
        priority=5,
    ),
    # Postal code (Korean 5-digit)
    PIIPattern(
        pii_type=PIIType.ADDRESS,
        pattern=r"\b(\d{5})\b(?=\s*[가-힣])",
        mask_template="[POSTAL_CODE_MASKED]",
        description="Korean postal code",
        priority=4,
    ),
    # Bank account numbers (Korean banks, various formats)
    PIIPattern(
        pii_type=PIIType.BANK_ACCOUNT,
        pattern=r"\b(\d{3,4})[-\s]?(\d{2,6})[-\s]?(\d{2,6})[-\s]?(\d{1,4})?\b",
        mask_template="[BANK_ACCOUNT_MASKED]",
        description="Bank account number",
        priority=3,
    ),
    # Credit card numbers (16 digits, various formats)
    PIIPattern(
        pii_type=PIIType.CREDIT_CARD,
        pattern=r"\b(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})\b",
        mask_template="[CREDIT_CARD_MASKED]",
        description="Credit card number",
        priority=9,
    ),
    # Vehicle license plates (Korean format)
    PIIPattern(
        pii_type=PIIType.VEHICLE_PLATE,
        pattern=r"\b(\d{2,3})\s*([가-힣])\s*(\d{4})\b",
        mask_template="[VEHICLE_PLATE_MASKED]",
        description="Korean vehicle license plate",
        priority=2,
    ),
    # IP addresses
    PIIPattern(
        pii_type=PIIType.IP_ADDRESS,
        pattern=r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        mask_template="[IP_ADDRESS_MASKED]",
        description="IP address",
        priority=1,
    ),
]

# Common Korean surnames (for name detection)
KOREAN_SURNAMES: Set[str] = {
    "김",
    "이",
    "박",
    "최",
    "정",
    "강",
    "조",
    "윤",
    "장",
    "임",
    "한",
    "오",
    "서",
    "신",
    "권",
    "황",
    "안",
    "송",
    "류",
    "전",
    "홍",
    "고",
    "문",
    "양",
    "손",
    "배",
    "백",
    "허",
    "유",
    "남",
    "심",
    "노",
    "하",
    "곽",
    "성",
    "차",
    "주",
    "우",
    "구",
    "민",
    "진",
    "지",
    "엄",
    "변",
    "추",
    "도",
    "소",
    "석",
    "선",
    "설",
}


class PIIMasker:
    """
    Personal Identifiable Information (PII) Masker

    Detects and masks various types of PII in Korean text data.
    """

    def __init__(
        self,
        patterns: Optional[List[PIIPattern]] = None,
        enabled_types: Optional[Set[PIIType]] = None,
        custom_mask_templates: Optional[Dict[PIIType, str]] = None,
    ):
        """
        Initialize the PII masker.

        Args:
            patterns: Custom PII patterns (uses default if None)
            enabled_types: Set of PII types to detect (all if None)
            custom_mask_templates: Custom mask templates for each type
        """
        self.patterns = patterns or PII_PATTERNS
        self.enabled_types = enabled_types or set(PIIType)
        self.custom_mask_templates = custom_mask_templates or {}

        # Sort patterns by priority (higher first)
        self.patterns = sorted(self.patterns, key=lambda p: p.priority, reverse=True)

        # Compile regex patterns
        self._compiled_patterns: List[Tuple[PIIPattern, re.Pattern]] = []
        self._compile_patterns()

        # Statistics
        self.stats: Dict[PIIType, int] = {pii_type: 0 for pii_type in PIIType}

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency"""
        for pattern in self.patterns:
            if pattern.pii_type in self.enabled_types:
                try:
                    compiled = re.compile(pattern.pattern, re.IGNORECASE | re.UNICODE)
                    self._compiled_patterns.append((pattern, compiled))
                except re.error as e:
                    logger.error(f"Failed to compile pattern {pattern.pii_type}: {e}")

    def _get_mask_template(self, pii_type: PIIType, pattern: PIIPattern) -> str:
        """Get mask template for a PII type"""
        if pii_type in self.custom_mask_templates:
            return self.custom_mask_templates[pii_type]
        return pattern.mask_template

    def detect_pii(self, text: str) -> List[PIIMatch]:
        """
        Detect all PII in text without masking.

        Args:
            text: Input text to analyze

        Returns:
            List of PIIMatch objects
        """
        if not text:
            return []

        matches: List[PIIMatch] = []
        masked_positions: Set[Tuple[int, int]] = set()

        for pattern, compiled in self._compiled_patterns:
            for match in compiled.finditer(text):
                start, end = match.span()

                # Skip if this position is already matched by a higher priority pattern
                overlap = False
                for pos_start, pos_end in masked_positions:
                    if not (end <= pos_start or start >= pos_end):
                        overlap = True
                        break

                if not overlap:
                    masked_positions.add((start, end))
                    matches.append(
                        PIIMatch(
                            pii_type=pattern.pii_type,
                            original_text=match.group(),
                            start_pos=start,
                            end_pos=end,
                            masked_text=self._get_mask_template(pattern.pii_type, pattern),
                        )
                    )

        return sorted(matches, key=lambda m: m.start_pos)

    def mask_text(self, text: str) -> str:
        """
        Mask all detected PII in text.

        Args:
            text: Input text

        Returns:
            Text with PII masked
        """
        if not text:
            return ""

        matches = self.detect_pii(text)

        if not matches:
            return text

        # Build masked text from end to start to preserve positions
        result = text
        for match in reversed(matches):
            result = result[: match.start_pos] + match.masked_text + result[match.end_pos :]
            self.stats[match.pii_type] += 1

        return result

    def mask_korean_name(self, text: str) -> str:
        """
        Mask Korean names in text using surname detection.

        This is a heuristic approach that looks for common Korean surnames
        followed by 1-2 Korean characters.

        Args:
            text: Input text

        Returns:
            Text with names masked
        """
        if not text:
            return ""

        result = text

        # Pattern: Korean surname + 1-2 Korean characters
        name_pattern = re.compile(
            r"([" + "".join(KOREAN_SURNAMES) + r"])([가-힣]{1,2})(?=\s|님|씨|$|[^가-힣])"
        )

        for match in name_pattern.finditer(text):
            full_name = match.group()
            masked = "[NAME_MASKED]"
            result = result.replace(full_name, masked, 1)
            self.stats[PIIType.NAME] += 1

        return result

    def mask_all(self, text: str, include_name_detection: bool = True) -> str:
        """
        Apply all masking strategies.

        Args:
            text: Input text
            include_name_detection: Whether to include heuristic name detection

        Returns:
            Fully masked text
        """
        if not text:
            return ""

        # First apply pattern-based masking
        result = self.mask_text(text)

        # Then apply name detection if enabled
        if include_name_detection:
            result = self.mask_korean_name(result)

        return result

    def get_statistics(self) -> Dict[str, int]:
        """Get masking statistics"""
        return {pii_type.value: count for pii_type, count in self.stats.items()}

    def reset_statistics(self) -> None:
        """Reset masking statistics"""
        self.stats = {pii_type: 0 for pii_type in PIIType}

    @classmethod
    def create_strict_masker(cls) -> "PIIMasker":
        """Create a masker with all PII types enabled"""
        return cls(enabled_types=set(PIIType))

    @classmethod
    def create_basic_masker(cls) -> "PIIMasker":
        """Create a masker with only essential PII types"""
        essential_types = {
            PIIType.RESIDENT_ID,
            PIIType.PHONE,
            PIIType.EMAIL,
        }
        return cls(enabled_types=essential_types)


def mask_pii(text: str, strict: bool = True) -> str:
    """
    Convenience function to mask PII in text.

    Args:
        text: Input text
        strict: If True, use strict masking (all PII types)

    Returns:
        Masked text
    """
    masker = PIIMasker.create_strict_masker() if strict else PIIMasker.create_basic_masker()
    return masker.mask_all(text)


def validate_no_pii(text: str) -> Tuple[bool, List[PIIMatch]]:
    """
    Validate that text contains no PII.

    Args:
        text: Text to validate

    Returns:
        Tuple of (is_clean, detected_matches)
    """
    masker = PIIMasker.create_strict_masker()
    matches = masker.detect_pii(text)
    return len(matches) == 0, matches


if __name__ == "__main__":
    # Test examples
    test_texts = [
        "Contact me at 010-1234-5678 or email@example.com",
        "My resident ID is 901231-1234567",
        "Address: Seoul Gangnam-gu Yeoksam-dong 123-45",
        "Kim Minsu submitted the complaint",
        "Please send the fee to account 123-456-789012",
        "Server IP is 192.168.1.100",
        "My car number is 12 ga 3456",
    ]

    masker = PIIMasker.create_strict_masker()

    print("PII Masking Test Results")
    print("=" * 60)

    for text in test_texts:
        masked = masker.mask_all(text)
        print(f"\nOriginal: {text}")
        print(f"Masked:   {masked}")

    print("\n" + "=" * 60)
    print("Statistics:")
    for pii_type, count in masker.get_statistics().items():
        if count > 0:
            print(f"  {pii_type}: {count}")
