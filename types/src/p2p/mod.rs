use std::net::{SocketAddr, SocketAddrV4, SocketAddrV6};

use multiaddr::Protocol;

use crate::{multiaddr::Multiaddr, peer_id::PeerId};

// pub mod connection_manager;
// pub mod network;
pub mod active_peers;
pub mod channel_manager;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerEvent {
    NewPeer {
        peer_id: PeerId,
        address: Multiaddr,
    },
    LostPeer {
        peer_id: PeerId,
        reason: DisconnectReason,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisconnectReason {
    ConnectionLost,
    Shutdown,
    RequestedDisconnect,
    SimultaneousDialResolution,
}

pub fn to_host_port_str(addr: &Multiaddr) -> Result<String, &'static str> {
    let mut iter = addr.iter();

    match (iter.next(), iter.next()) {
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Dns(hostname)), Some(Protocol::Tcp(port))) => {
            Ok(format!("{}:{}", hostname, port))
        }

        _ => {
            tracing::warn!("unsupported multiaddr: '{addr}'");
            Err("invalid address")
        }
    }
}

/// Attempts to convert a multiaddr of the form `/[ip4,ip6]/{}/[udp,tcp]/{port}` into
/// a SocketAddr value.
pub fn to_socket_addr(addr: &Multiaddr) -> Result<SocketAddr, &'static str> {
    let mut iter = addr.iter();

    match (iter.next(), iter.next()) {
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Udp(port)))
        | (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(SocketAddr::V4(SocketAddrV4::new(ipaddr, port)))
        }

        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Udp(port)))
        | (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(SocketAddr::V6(SocketAddrV6::new(ipaddr, port, 0, 0)))
        }

        _ => {
            tracing::warn!("unsupported multiaddr: '{addr}'");
            Err("invalid address")
        }
    }
}
