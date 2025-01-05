from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import jwt
import datetime
from typing import Dict, Any, Optional
import base64

class SecurityManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self._init_encryption_key()
        
    def _init_encryption_key(self) -> None:
        """암호화 키 초기화"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=os.urandom(16),
            iterations=100000,
        )
        self.encryption_key = base64.urlsafe_b64encode(kdf.derive(self.secret_key.encode()))
        self.fernet = Fernet(self.encryption_key)
        
    def encrypt_model(self, model_path: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """모델 파일 암호화"""
        with open(model_path, 'rb') as f:
            model_data = f.read()
            
        # 1. 메타데이터 추가
        metadata['encryption_timestamp'] = datetime.datetime.now().isoformat()
        metadata_bytes = json.dumps(metadata).encode()
        
        # 2. 모델 데이터와 메타데이터 결합
        combined_data = len(metadata_bytes).to_bytes(4, 'big') + metadata_bytes + model_data
        
        # 3. AES 암호화
        iv = os.urandom(16)
        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.CBC(iv)
        )
        encryptor = cipher.encryptor()
        
        # PKCS7 패딩
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(combined_data) + padder.finalize()
        
        # 암호화 수행
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # 4. IV와 암호화된 데이터 결합
        final_data = iv + encrypted_data
        
        # 5. 암호화된 파일 저장
        encrypted_path = f"{model_path}.encrypted"
        with open(encrypted_path, 'wb') as f:
            f.write(final_data)
            
        # 6. 복호화 키 생성
        decryption_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        
        return encrypted_path, decryption_key
        
    def decrypt_model(self, encrypted_path: str, decryption_key: str) -> Tuple[bytes, Dict[str, Any]]:
        """암호화된 모델 파일 복호화"""
        with open(encrypted_path, 'rb') as f:
            data = f.read()
            
        # 1. IV 추출
        iv = data[:16]
        encrypted_data = data[16:]
        
        # 2. AES 복호화
        cipher = Cipher(
            algorithms.AES(base64.urlsafe_b64decode(decryption_key)),
            modes.CBC(iv)
        )
        decryptor = cipher.decryptor()
        
        # 복호화 수행
        decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # PKCS7 언패딩
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
        
        # 3. 메타데이터 분리
        metadata_length = int.from_bytes(unpadded_data[:4], 'big')
        metadata_bytes = unpadded_data[4:4+metadata_length]
        model_data = unpadded_data[4+metadata_length:]
        
        # 4. 메타데이터 파싱
        metadata = json.loads(metadata_bytes.decode())
        
        return model_data, metadata
        
    def generate_model_token(self, user_id: int, model_id: str, subscription_type: str) -> str:
        """모델 액세스 토큰 생성"""
        now = datetime.datetime.utcnow()
        
        # 구독 타입에 따른 토큰 유효기간 설정
        if subscription_type == 'premium':
            expire_delta = datetime.timedelta(days=30)
        else:
            expire_delta = datetime.timedelta(days=7)
            
        payload = {
            'user_id': user_id,
            'model_id': model_id,
            'subscription_type': subscription_type,
            'iat': now,
            'exp': now + expire_delta,
            'offline_access_limit': subscription_type == 'premium'
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
        
    def verify_model_token(self, token: str) -> Optional[Dict[str, Any]