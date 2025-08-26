"""
TLS 1.3 Certificate Management and Encryption

Provides secure TLS 1.3 encryption for all data transmission with
automated certificate management, renewal, and validation.
"""

import ssl
import asyncio
import logging
import ipaddress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import aiofiles

logger = logging.getLogger(__name__)


class TLSManager:
    """Manages TLS 1.3 certificates and secure connections"""
    
    def __init__(self, cert_dir: str = "certs", auto_renew: bool = True):
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(exist_ok=True)
        self.auto_renew = auto_renew
        self.certificates: Dict[str, Dict] = {}
        
    async def generate_self_signed_cert(
        self,
        hostname: str,
        key_size: int = 2048,
        validity_days: int = 365
    ) -> Tuple[str, str]:
        """Generate self-signed certificate for development/testing"""
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ESCAI Framework"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName(f"*.{hostname}"),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256(), default_backend())
        
        # Save certificate and key
        cert_path = self.cert_dir / f"{hostname}.crt"
        key_path = self.cert_dir / f"{hostname}.key"
        
        async with aiofiles.open(cert_path, 'wb') as f:
            await f.write(cert.public_bytes(serialization.Encoding.PEM))
            
        async with aiofiles.open(key_path, 'wb') as f:
            await f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        logger.info(f"Generated self-signed certificate for {hostname}")
        return str(cert_path), str(key_path)
    
    async def load_certificate(self, hostname: str) -> Optional[Dict]:
        """Load certificate information for hostname"""
        cert_path = self.cert_dir / f"{hostname}.crt"
        key_path = self.cert_dir / f"{hostname}.key"
        
        if not cert_path.exists() or not key_path.exists():
            return None
            
        try:
            async with aiofiles.open(cert_path, 'rb') as f:
                cert_data = await f.read()
                cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            
            return {
                'cert_path': str(cert_path),
                'key_path': str(key_path),
                'not_valid_before': cert.not_valid_before,
                'not_valid_after': cert.not_valid_after,
                'subject': cert.subject.rfc4514_string(),
                'issuer': cert.issuer.rfc4514_string(),
                'serial_number': str(cert.serial_number)
            }
        except Exception as e:
            logger.error(f"Failed to load certificate for {hostname}: {e}")
            return None
    
    async def is_certificate_valid(self, hostname: str) -> bool:
        """Check if certificate is valid and not expired"""
        cert_info = await self.load_certificate(hostname)
        if not cert_info:
            return False
            
        now = datetime.utcnow()
        return (cert_info['not_valid_before'] <= now <= cert_info['not_valid_after'])
    
    async def needs_renewal(self, hostname: str, days_before_expiry: int = 30) -> bool:
        """Check if certificate needs renewal"""
        cert_info = await self.load_certificate(hostname)
        if not cert_info:
            return True
            
        expiry_threshold = datetime.utcnow() + timedelta(days=days_before_expiry)
        return cert_info['not_valid_after'] <= expiry_threshold
    
    async def renew_certificate(self, hostname: str) -> bool:
        """Renew certificate if needed"""
        try:
            if await self.needs_renewal(hostname):
                await self.generate_self_signed_cert(hostname)
                logger.info(f"Renewed certificate for {hostname}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to renew certificate for {hostname}: {e}")
            return False
    
    def create_ssl_context(
        self,
        cert_path: str,
        key_path: str,
        client_auth: bool = False
    ) -> ssl.SSLContext:
        """Create SSL context with TLS 1.3 configuration"""
        
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # Force TLS 1.3
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Load certificate and key
        context.load_cert_chain(cert_path, key_path)
        
        # Security settings
        context.set_ciphers('TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256')
        context.check_hostname = False  # We handle hostname verification separately
        context.verify_mode = ssl.CERT_REQUIRED if client_auth else ssl.CERT_NONE
        
        # Additional security options
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        context.options |= ssl.OP_NO_TLSv1_2
        context.options |= ssl.OP_SINGLE_DH_USE
        context.options |= ssl.OP_SINGLE_ECDH_USE
        
        return context
    
    async def setup_server_ssl(self, hostname: str) -> ssl.SSLContext:
        """Setup SSL context for server with automatic certificate management"""
        
        # Check if certificate exists and is valid
        if not await self.is_certificate_valid(hostname):
            logger.info(f"Generating new certificate for {hostname}")
            await self.generate_self_signed_cert(hostname)
        
        # Auto-renew if needed
        if self.auto_renew:
            await self.renew_certificate(hostname)
        
        cert_info = await self.load_certificate(hostname)
        if not cert_info:
            raise ValueError(f"Failed to setup SSL for {hostname}")
        
        return self.create_ssl_context(
            cert_info['cert_path'],
            cert_info['key_path']
        )
    
    async def verify_peer_certificate(self, peer_cert: bytes, hostname: str) -> bool:
        """Verify peer certificate against expected hostname"""
        try:
            cert = x509.load_pem_x509_certificate(peer_cert, default_backend())
            
            # Check if certificate is not expired
            now = datetime.utcnow()
            if not (cert.not_valid_before <= now <= cert.not_valid_after):
                return False
            
            # Check hostname in SAN extension
            try:
                san_ext = cert.extensions.get_extension_for_oid(
                    x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
                ).value
                
                for name in san_ext:
                    if isinstance(name, x509.DNSName) and name.value == hostname:
                        return True
            except x509.ExtensionNotFound:
                pass
            
            # Check common name
            try:
                common_name = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
                return common_name == hostname
            except (IndexError, AttributeError):
                pass
            
            return False
            
        except Exception as e:
            logger.error(f"Certificate verification failed: {e}")
            return False
    
    async def start_auto_renewal_task(self):
        """Start background task for automatic certificate renewal"""
        if not self.auto_renew:
            return
            
        async def renewal_task():
            while True:
                try:
                    # Check all certificates every 24 hours
                    for cert_file in self.cert_dir.glob("*.crt"):
                        hostname = cert_file.stem
                        await self.renew_certificate(hostname)
                    
                    await asyncio.sleep(86400)  # 24 hours
                except Exception as e:
                    logger.error(f"Auto-renewal task error: {e}")
                    await asyncio.sleep(3600)  # Retry in 1 hour
        
        asyncio.create_task(renewal_task())
        logger.info("Started automatic certificate renewal task")