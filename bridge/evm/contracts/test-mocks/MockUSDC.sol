// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title MockUSDC
/// @notice Minimal mintable ERC20 stand-in for USDC, used by both the
/// Foundry tests and the Rust `bridge-node`'s anvil end-to-end test.
/// 6 decimals (USDC parity); no allowance/approval gas optimizations.
///
/// NOT for production — `mint` is unauthorized so any test wallet can
/// fund itself.
contract MockUSDC {
    string public constant name = "USD Coin (Mock)";
    string public constant symbol = "USDC";
    uint8 public constant decimals = 6;

    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply += amount;
        emit Transfer(address(0), to, amount);
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "USDC: insufficient");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(balanceOf[from] >= amount, "USDC: insufficient");
        require(allowance[from][msg.sender] >= amount, "USDC: not approved");
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        emit Transfer(from, to, amount);
        return true;
    }
}
