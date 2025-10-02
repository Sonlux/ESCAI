"""
PII Detection and Masking System

Provides comprehensive PII detection and masking capabilities with
configurable sensitivity levels and pattern-based detection.
"""

import re
import hashlib
import secrets
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    API_KEY = "api_key"
    PASSWORD = "password"  # nosec B105
    CUSTOM = "custom"


class SensitivityLevel(Enum):
    """Sensitivity levels for PII detection"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    STRICT = 4


@dataclass
class PIIMatch:
    """Represents a detected PII match"""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str = ""


@dataclass
class MaskingRule:
    """Rule for masking specific PII types"""
    pii_type: PIIType
    mask_char: str = "*"
    preserve_chars: int = 0  # Number of chars to preserve at start/end
    hash_instead: bool = False
    replacement_pattern: Optional[str] = None


class PIIDetector:
    """Detects various types of PII in text"""
    
    def __init__(self, sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM):
        self.sensitivity_level = sensitivity_level
        self.patterns = self._initialize_patterns()
        self.custom_patterns: Dict[str, str] = {}
        
    def _initialize_patterns(self) -> Dict[PIIType, List[str]]:
        """Initialize regex patterns for PII detection"""
        patterns = {
            PIIType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            PIIType.PHONE: [
                r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
                r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}'
            ],
            PIIType.SSN: [
                r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
                r'\b\d{9}\b'
            ],
            PIIType.CREDIT_CARD: [
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
            ],
            PIIType.IP_ADDRESS: [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
            ],
            PIIType.DATE_OF_BIRTH: [
                r'\b(?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12][0-9]|3[01])[-/.](?:19|20)\d{2}\b',
                r'\b(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12][0-9]|3[01])\b'
            ],
            PIIType.PASSPORT: [
                r'\b[A-Z]{1,2}[0-9]{6,9}\b'
            ],
            PIIType.DRIVER_LICENSE: [
                r'\b[A-Z]{1,2}[0-9]{6,8}\b'
            ],
            PIIType.BANK_ACCOUNT: [
                r'\b[0-9]{8,17}\b'
            ],
            PIIType.API_KEY: [
                r'\b[A-Za-z0-9]{32,}\b',
                r'api[_-]?key["\s]*[:=]["\s]*[A-Za-z0-9]+',
                r'secret[_-]?key["\s]*[:=]["\s]*[A-Za-z0-9]+',
                r'access[_-]?token["\s]*[:=]["\s]*[A-Za-z0-9]+'
            ],
            PIIType.PASSWORD: [
                r'password["\s]*[:=]["\s]*[^\s"]+',
                r'passwd["\s]*[:=]["\s]*[^\s"]+',
                r'pwd["\s]*[:=]["\s]*[^\s"]+'
            ]
        }
        
        # Add name patterns based on sensitivity level
        if self.sensitivity_level.value >= SensitivityLevel.MEDIUM.value:
            patterns[PIIType.NAME] = [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b[A-Z][a-z]+, [A-Z][a-z]+\b'  # Last, First
            ]
        
        # Add address patterns for higher sensitivity
        if self.sensitivity_level.value >= SensitivityLevel.HIGH.value:
            patterns[PIIType.ADDRESS] = [
                r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
                r'\b[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'
            ]
        
        return patterns
    
    def add_custom_pattern(self, name: str, pattern: str, pii_type: PIIType = PIIType.CUSTOM):
        """Add custom PII detection pattern"""
        self.custom_patterns[name] = pattern
        if pii_type not in self.patterns:
            self.patterns[pii_type] = []
        self.patterns[pii_type].append(pattern)
    
    def detect_pii(self, text: str, context: str = "") -> List[PIIMatch]:
        """Detect PII in text and return matches"""
        matches = []
        
        for pii_type, patterns in self.patterns.items():
            for pattern in patterns:
                try:
                    regex_matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in regex_matches:
                        confidence = self._calculate_confidence(pii_type, match.group())
                        
                        # Filter by confidence based on sensitivity level
                        min_confidence = self._get_min_confidence()
                        if confidence >= min_confidence:
                            matches.append(PIIMatch(
                                pii_type=pii_type,
                                value=match.group(),
                                start_pos=match.start(),
                                end_pos=match.end(),
                                confidence=confidence,
                                context=context
                            ))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for {pii_type}: {e}")
        
        # Remove overlapping matches (keep highest confidence)
        return self._remove_overlapping_matches(matches)
    
    def _calculate_confidence(self, pii_type: PIIType, value: str) -> float:
        """Calculate confidence score for PII match"""
        base_confidence = 0.7
        
        # Adjust confidence based on PII type
        type_adjustments = {
            PIIType.EMAIL: 0.9,
            PIIType.PHONE: 0.8,
            PIIType.SSN: 0.95,
            PIIType.CREDIT_CARD: 0.9,
            PIIType.IP_ADDRESS: 0.7,
            PIIType.NAME: 0.6,
            PIIType.ADDRESS: 0.7,
            PIIType.DATE_OF_BIRTH: 0.8,
            PIIType.API_KEY: 0.8,
            PIIType.PASSWORD: 0.9
        }
        
        confidence = type_adjustments.get(pii_type, base_confidence)
        
        # Additional validation for specific types
        if pii_type == PIIType.CREDIT_CARD:
            confidence *= self._validate_credit_card(value)
        elif pii_type == PIIType.SSN:
            confidence *= self._validate_ssn(value)
        elif pii_type == PIIType.EMAIL:
            confidence *= self._validate_email(value)
        
        return min(confidence, 1.0)
    
    def _validate_credit_card(self, number: str) -> float:
        """Validate credit card using Luhn algorithm"""
        digits = [int(d) for d in re.sub(r'\D', '', number)]
        if len(digits) < 13 or len(digits) > 19:
            return 0.5
        
        # Luhn algorithm
        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        
        return 1.0 if checksum % 10 == 0 else 0.3
    
    def _validate_ssn(self, ssn: str) -> float:
        """Basic SSN validation"""
        digits = re.sub(r'\D', '', ssn)
        if len(digits) != 9:
            return 0.3
        
        # Check for invalid patterns
        if digits == '000000000' or digits[:3] == '000' or digits[3:5] == '00' or digits[5:] == '0000':
            return 0.1
        
        return 1.0
    
    def _validate_email(self, email: str) -> float:
        """Enhanced email validation"""
        if '@' not in email or email.count('@') != 1:
            return 0.3
        
        local, domain = email.split('@')
        if not local or not domain or '.' not in domain:
            return 0.5
        
        return 1.0
    
    def _get_min_confidence(self) -> float:
        """Get minimum confidence threshold based on sensitivity level"""
        thresholds = {
            SensitivityLevel.LOW: 0.9,
            SensitivityLevel.MEDIUM: 0.7,
            SensitivityLevel.HIGH: 0.5,
            SensitivityLevel.STRICT: 0.3
        }
        return thresholds[self.sensitivity_level]
    
    def _remove_overlapping_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence"""
        if not matches:
            return matches
        
        # Sort by position
        matches.sort(key=lambda m: m.start_pos)
        
        filtered_matches: List[PIIMatch] = []
        for match in matches:
            # Check if this match overlaps with any existing match
            overlaps = False
            for existing in filtered_matches:
                if (match.start_pos < existing.end_pos and 
                    match.end_pos > existing.start_pos):
                    # Overlapping - keep the one with higher confidence
                    if match.confidence > existing.confidence:
                        filtered_matches.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered_matches.append(match)
        
        return filtered_matches


class PIIMasker:
    """Masks detected PII in text"""
    
    def __init__(self) -> None:
        self.masking_rules: Dict[PIIType, MaskingRule] = self._initialize_default_rules()
        self.hash_salt = secrets.token_hex(16)
    
    def _initialize_default_rules(self) -> Dict[PIIType, MaskingRule]:
        """Initialize default masking rules"""
        return {
            PIIType.EMAIL: MaskingRule(PIIType.EMAIL, preserve_chars=2),
            PIIType.PHONE: MaskingRule(PIIType.PHONE, preserve_chars=2),
            PIIType.SSN: MaskingRule(PIIType.SSN, preserve_chars=0),
            PIIType.CREDIT_CARD: MaskingRule(PIIType.CREDIT_CARD, preserve_chars=4),
            PIIType.IP_ADDRESS: MaskingRule(PIIType.IP_ADDRESS, hash_instead=True),
            PIIType.NAME: MaskingRule(PIIType.NAME, replacement_pattern="[NAME]"),
            PIIType.ADDRESS: MaskingRule(PIIType.ADDRESS, replacement_pattern="[ADDRESS]"),
            PIIType.DATE_OF_BIRTH: MaskingRule(PIIType.DATE_OF_BIRTH, replacement_pattern="[DOB]"),
            PIIType.API_KEY: MaskingRule(PIIType.API_KEY, preserve_chars=4, hash_instead=True),
            PIIType.PASSWORD: MaskingRule(PIIType.PASSWORD, replacement_pattern="[REDACTED]")
        }
    
    def set_masking_rule(self, pii_type: PIIType, rule: MaskingRule):
        """Set custom masking rule for PII type"""
        self.masking_rules[pii_type] = rule
    
    def mask_text(self, text: str, pii_matches: List[PIIMatch]) -> str:
        """Mask PII in text based on detected matches"""
        if not pii_matches:
            return text
        
        # Sort matches by position (reverse order to maintain positions)
        sorted_matches = sorted(pii_matches, key=lambda m: m.start_pos, reverse=True)
        
        masked_text = text
        for match in sorted_matches:
            masked_value = self._mask_value(match)
            masked_text = (
                masked_text[:match.start_pos] + 
                masked_value + 
                masked_text[match.end_pos:]
            )
        
        return masked_text
    
    def _mask_value(self, match: PIIMatch) -> str:
        """Mask individual PII value"""
        rule = self.masking_rules.get(match.pii_type)
        if not rule:
            # Default masking
            return "*" * len(match.value)
        
        if rule.replacement_pattern:
            return rule.replacement_pattern
        
        if rule.hash_instead:
            return self._hash_value(match.value)
        
        return self._apply_character_masking(match.value, rule)
    
    def _hash_value(self, value: str) -> str:
        """Create consistent hash of PII value"""
        hash_input = f"{value}{self.hash_salt}".encode()
        hash_hex = hashlib.sha256(hash_input).hexdigest()
        return f"[HASH:{hash_hex[:8]}]"
    
    def _apply_character_masking(self, value: str, rule: MaskingRule) -> str:
        """Apply character-based masking"""
        if rule.preserve_chars == 0:
            return rule.mask_char * len(value)
        
        if len(value) <= rule.preserve_chars * 2:
            return rule.mask_char * len(value)
        
        preserve_start = value[:rule.preserve_chars]
        preserve_end = value[-rule.preserve_chars:] if rule.preserve_chars > 0 else ""
        middle_length = len(value) - (rule.preserve_chars * 2)
        
        return preserve_start + (rule.mask_char * middle_length) + preserve_end
    
    def mask_structured_data(self, data: Dict[str, Any], detector: PIIDetector) -> Dict[str, Any]:
        """Mask PII in structured data (dictionaries)"""
        if not isinstance(data, dict):
            return data
        
        masked_data: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                pii_matches = detector.detect_pii(value, context=key)
                masked_data[key] = self.mask_text(value, pii_matches)
            elif isinstance(value, dict):
                masked_data[key] = self.mask_structured_data(value, detector)
            elif isinstance(value, list):
                masked_data[key] = [
                    self.mask_structured_data(item, detector) if isinstance(item, dict)
                    else self.mask_text(str(item), detector.detect_pii(str(item), context=key)) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                masked_data[key] = value
        
        return masked_data
    
    def get_masking_summary(self, pii_matches: List[PIIMatch]) -> Dict[str, int]:
        """Get summary of masked PII types"""
        summary: Dict[str, int] = {}
        for match in pii_matches:
            pii_type_name = match.pii_type.value
            summary[pii_type_name] = summary.get(pii_type_name, 0) + 1
        return summary