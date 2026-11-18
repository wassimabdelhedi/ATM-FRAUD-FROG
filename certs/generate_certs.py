"""Generateur de certificats TLS auto-signes pour le reseau Fog ATM.

Cree :
  certs/ca.crt + ca.key        -- Autorite de Certification interne (Root CA)
  certs/fog_a.crt + fog_a.key  -- Certificat du noeud Fog A
  certs/fog_b.crt + fog_b.key  -- Certificat du noeud Fog B
  certs/fog_c.crt + fog_c.key  -- Certificat du noeud Fog C

Utilisation (a executer une seule fois depuis la racine du projet) :
  ml\\venv\\Scripts\\python.exe certs\\generate_certs.py

NOTE : Dans un contexte bancaire reel, les certificats seraient emis par
       une PKI interne de la banque (ex: Microsoft ADCS, HashiCorp Vault).
       Ces certificats auto-signes sont utilises uniquement pour le prototype.
"""
import datetime
import ipaddress
import os

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

OUT_DIR = os.path.dirname(__file__)  # dossier certs/
VALIDITY_DAYS = 3650                 # 10 ans pour le prototype

NODES = ["fog_a", "fog_b", "fog_c"]
# SAN : chaque cert est valide pour localhost ET l'IP 127.0.0.1
# (pour le deploiement LAN, ajoutez l'IP de chaque PC ici)
SANS = [
    x509.DNSName("localhost"),
    x509.DNSName("fog_a"),
    x509.DNSName("fog_b"),
    x509.DNSName("fog_c"),
    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
]


def _save(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)
    print(f"  Ecrit : {path}")


def _gen_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _key_pem(key: rsa.RSAPrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _cert_pem(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


def generate():
    now = datetime.datetime.now(datetime.timezone.utc)
    expiry = now + datetime.timedelta(days=VALIDITY_DAYS)

    # ------------------------------------------------------------------
    # 1. Autorite de Certification (Root CA)
    # ------------------------------------------------------------------
    print("Generation du Root CA...")
    ca_key = _gen_key()
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "TN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Fog-ATM-Bank"),
        x509.NameAttribute(NameOID.COMMON_NAME, "Fog-ATM-Root-CA"),
    ])
    ca_builder = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(expiry)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
    )
    # SubjectKeyIdentifier + AuthorityKeyIdentifier obligatoires pour Python ssl
    ca_ski = x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key())
    ca_builder = ca_builder.add_extension(ca_ski, critical=False)
    ca_builder = ca_builder.add_extension(
        x509.AuthorityKeyIdentifier(
            key_identifier=ca_ski.digest,
            authority_cert_issuer=None,
            authority_cert_serial_number=None,
        ),
        critical=False,
    )
    ca_cert = ca_builder.sign(ca_key, hashes.SHA256())
    _save(os.path.join(OUT_DIR, "ca.key"), _key_pem(ca_key))
    _save(os.path.join(OUT_DIR, "ca.crt"), _cert_pem(ca_cert))

    # ------------------------------------------------------------------
    # 2. Certificat pour chaque noeud Fog (signe par le CA)
    # ------------------------------------------------------------------
    for node in NODES:
        print(f"\nGeneration du certificat pour {node}...")
        node_key = _gen_key()
        node_name = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "TN"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Fog-ATM-Bank"),
            x509.NameAttribute(NameOID.COMMON_NAME, node),
        ])
        node_cert = (
            x509.CertificateBuilder()
            .subject_name(node_name)
            .issuer_name(ca_name)
            .public_key(node_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(expiry)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.SubjectAlternativeName(SANS),
                critical=False,
            )
            .add_extension(
                x509.ExtendedKeyUsage([
                    x509.ExtendedKeyUsageOID.SERVER_AUTH,
                    x509.ExtendedKeyUsageOID.CLIENT_AUTH,  # Pour mTLS
                ]),
                critical=False,
            )
            # SubjectKeyIdentifier + AuthorityKeyIdentifier (requis par Python ssl)
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(node_key.public_key()),
                critical=False,
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256())  # Signe par le CA
        )
        _save(os.path.join(OUT_DIR, f"{node}.key"), _key_pem(node_key))
        _save(os.path.join(OUT_DIR, f"{node}.crt"), _cert_pem(node_cert))

    print("\nCertificats generes avec succes !")
    print("Contenu du dossier certs/ :")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith((".crt", ".key")):
            path = os.path.join(OUT_DIR, f)
            size = os.path.getsize(path)
            print(f"  {f:<20} {size:>6} octets")


if __name__ == "__main__":
    generate()
