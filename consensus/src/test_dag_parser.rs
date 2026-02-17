// Portions of this file are derived from Mysticeti consensus (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/consensus/core/src/test_dag_parser.rs
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use crate::test_dag_builder::DagBuilder;
use types::committee::AuthorityIndex;

use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, take_while_m_n, take_while1},
    character::complete::{char, digit1, multispace0, multispace1, space0, space1},
    combinator::{map_res, opt},
    multi::{many0, separated_list0},
    sequence::{delimited, preceded, terminated, tuple},
};
use types::consensus::{
    block::{BlockRef, Round, Slot},
    context::Context,
};

/// DagParser
///
/// Usage:
///
/// ```
/// let dag_str = "DAG {
///     Round 0 : { 4 },
///     Round 1 : { * },
///     Round 2 : { * },
///     Round 3 : { * },
///     Round 4 : {
///         A -> [-D3],
///         B -> [*],
///         C -> [*],
///         D -> [*],
///     },
///     Round 5 : {
///         A -> [*],
///         B -> [*],
///         C -> [A4],
///         D -> [A4],
///     },
///     Round 6 : { * },
///     Round 7 : { * },
///     Round 8 : { * },
///     }";
///
/// let (_, dag_builder) = parse_dag(dag_str).expect("Invalid dag"); // parse DAG DSL
/// dag_builder.print(); // print the parsed DAG
/// dag_builder.persist_all_blocks(dag_state.clone()); // persist all blocks to DagState
/// ```
pub(crate) fn parse_dag(dag_string: &str) -> IResult<&str, DagBuilder> {
    let (input, _) = tuple((tag("DAG"), multispace0, char('{')))(dag_string)?;

    let (mut input, num_authors) = parse_genesis(input)?;

    let context = Arc::new(Context::new_for_test(num_authors as usize).0);
    let mut dag_builder = DagBuilder::new(context);

    // Parse subsequent rounds
    loop {
        match parse_round(input, &dag_builder) {
            Ok((new_input, (round, connections))) => {
                dag_builder.layer_with_connections(connections, round);
                input = new_input
            }
            Err(nom::Err::Error(_)) | Err(nom::Err::Failure(_)) => break,
            Err(nom::Err::Incomplete(needed)) => return Err(nom::Err::Incomplete(needed)),
        }
    }
    let (input, _) = tuple((multispace0, char('}')))(input)?;

    Ok((input, dag_builder))
}

#[allow(clippy::type_complexity)]
fn parse_round<'a>(
    input: &'a str,
    dag_builder: &DagBuilder,
) -> IResult<&'a str, (Round, Vec<(AuthorityIndex, Vec<BlockRef>)>)> {
    let (input, _) = tuple((multispace0, tag("Round"), space1))(input)?;
    let (input, round) = take_while1(|c: char| c.is_ascii_digit())(input)?;

    let (input, connections) = alt((
        |input| parse_fully_connected(input, dag_builder),
        |input| parse_specified_connections(input, dag_builder),
    ))(input)?;

    Ok((input, (round.parse().unwrap(), connections)))
}

fn parse_fully_connected<'a>(
    input: &'a str,
    dag_builder: &DagBuilder,
) -> IResult<&'a str, Vec<(AuthorityIndex, Vec<BlockRef>)>> {
    let (input, _) = tuple((
        space0,
        char(':'),
        space0,
        char('{'),
        space0,
        char('*'),
        space0,
        char('}'),
        opt(char(',')),
    ))(input)?;

    let ancestors = dag_builder.last_ancestors.clone();
    let connections = dag_builder
        .context
        .committee
        .authorities()
        .map(|authority| (authority.0, ancestors.clone()))
        .collect::<Vec<_>>();

    Ok((input, connections))
}

fn parse_specified_connections<'a>(
    input: &'a str,
    dag_builder: &DagBuilder,
) -> IResult<&'a str, Vec<(AuthorityIndex, Vec<BlockRef>)>> {
    let (input, _) = tuple((space0, char(':'), space0, char('{'), multispace0))(input)?;

    // parse specified connections
    // case 1: all authorities; [*]
    // case 2: specific included authorities; [A0, B0, C0]
    // case 3: specific excluded authorities;  [-A0]
    // case 4: mixed all authorities + specific included/excluded authorities; [*, A0]
    // TODO: case 5: byzantine case of multiple blocks per slot; [*]; timestamp=1
    let (input, authors_and_connections) = many0(parse_author_and_connections)(input)?;

    let mut output = Vec::new();
    for (author, connections) in authors_and_connections {
        let mut block_refs = HashSet::new();
        for connection in connections {
            if connection == "*" {
                block_refs.extend(dag_builder.last_ancestors.clone());
            } else if connection.starts_with('-') {
                let (input, _) = char('-')(connection)?;
                let (_, slot) = parse_slot(input)?;
                let stored_block_refs = get_blocks(slot, dag_builder);
                block_refs.extend(dag_builder.last_ancestors.clone());

                block_refs.retain(|ancestor| !stored_block_refs.contains(ancestor));
            } else {
                let input = connection;
                let (_, slot) = parse_slot(input)?;
                let stored_block_refs = get_blocks(slot, dag_builder);

                block_refs.extend(stored_block_refs);
            }
        }
        output.push((author, block_refs.into_iter().collect()));
    }

    let (input, _) = tuple((multispace0, char('}'), opt(char(','))))(input)?;

    Ok((input, output))
}

fn get_blocks(slot: Slot, dag_builder: &DagBuilder) -> Vec<BlockRef> {
    // note: special case for genesis blocks as they are cached separately

    if slot.round == 0 {
        dag_builder
            .genesis_block_refs()
            .into_iter()
            .filter(|block| Slot::from(*block) == slot)
            .collect::<Vec<_>>()
    } else {
        dag_builder
            .get_uncommitted_blocks_at_slot(slot)
            .iter()
            .map(|block| block.reference())
            .collect::<Vec<_>>()
    }
}

fn parse_author_and_connections(input: &str) -> IResult<&str, (AuthorityIndex, Vec<&str>)> {
    // parse author
    let (input, author) = preceded(
        multispace0,
        terminated(take_while1(|c: char| c.is_alphabetic()), preceded(opt(space0), tag("->"))),
    )(input)?;

    // parse connections
    let (input, connections) = delimited(
        preceded(opt(space0), char('[')),
        separated_list0(tag(", "), parse_block),
        terminated(char(']'), opt(char(','))),
    )(input)?;
    let (input, _) = opt(multispace1)(input)?;
    Ok((input, (str_to_authority_index(author).expect("Invalid authority index"), connections)))
}

fn parse_block(input: &str) -> IResult<&str, &str> {
    alt((
        map_res(tag("*"), |s: &str| Ok::<_, nom::error::ErrorKind>(s)),
        map_res(take_while1(|c: char| c.is_alphanumeric() || c == '-'), |s: &str| {
            Ok::<_, nom::error::ErrorKind>(s)
        }),
    ))(input)
}

fn parse_genesis(input: &str) -> IResult<&str, u32> {
    let (input, num_authorities) = preceded(
        tuple((
            multispace0,
            tag("Round"),
            space1,
            char('0'),
            space0,
            char(':'),
            space0,
            char('{'),
            space0,
        )),
        |i| parse_authority_count(i),
    )(input)?;
    let (input, _) = tuple((space0, char('}'), opt(char(','))))(input)?;

    Ok((input, num_authorities))
}

fn parse_authority_count(input: &str) -> IResult<&str, u32> {
    let (input, num_str) = digit1(input)?;
    Ok((input, num_str.parse().unwrap()))
}

fn parse_slot(input: &str) -> IResult<&str, Slot> {
    let parse_authority = map_res(
        take_while_m_n(1, 1, |c: char| c.is_alphabetic() && c.is_uppercase()),
        |letter: &str| {
            Ok::<_, nom::error::ErrorKind>(
                str_to_authority_index(letter).expect("Invalid authority index"),
            )
        },
    );

    let parse_round = map_res(digit1, |digits: &str| digits.parse::<Round>());

    let mut parser = tuple((parse_authority, parse_round));

    let (input, (authority, round)) = parser(input)?;
    Ok((input, Slot::new(round, authority)))
}

// Helper function to convert a string representation (e.g., 'A' or '[26]') to an AuthorityIndex
fn str_to_authority_index(input: &str) -> Option<AuthorityIndex> {
    if input.starts_with('[') && input.ends_with(']') && input.len() > 2 {
        input[1..input.len() - 1].parse::<u32>().ok().map(AuthorityIndex::new_for_test)
    } else if input.len() == 1 {
        let alpha_char = input.chars().next()?;
        if !alpha_char.is_ascii_uppercase() {
            return None;
        }
        let index = alpha_char as u32 - 'A' as u32;
        Some(AuthorityIndex::new_for_test(index))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::consensus::block::BlockAPI;

    #[tokio::test]
    async fn test_dag_parsing() {
        let dag_str = "DAG {
            Round 0 : { 4 },
            Round 1 : { * },
            Round 2 : { * },
            Round 3 : { * },
            Round 4 : {
                A -> [-D3],
                B -> [*],
                C -> [*],
                D -> [*],
            },
            Round 5 : {
                A -> [*],
                B -> [*],
                C -> [A4],
                D -> [A4],
            },
            Round 6 : { * },
        }";

        let (_, dag_builder) = parse_dag(dag_str).expect("Invalid dag");

        // Verify genesis has 4 authorities
        let genesis_refs = dag_builder.genesis_block_refs();
        assert_eq!(genesis_refs.len(), 4, "Expected 4 genesis blocks");

        // Verify blocks were created for rounds 1 through 6
        let all_blocks = dag_builder.blocks(1..=6);
        assert!(!all_blocks.is_empty(), "Expected blocks to be created");

        // Verify round 4 block A excludes D3: should have 3 ancestors (A3, B3, C3)
        let slot_a4 = Slot::new(4, AuthorityIndex::new_for_test(0));
        let a4_blocks = dag_builder.get_uncommitted_blocks_at_slot(slot_a4);
        assert_eq!(a4_blocks.len(), 1, "Expected exactly one block at A4");
        let a4_block = &a4_blocks[0];
        assert_eq!(a4_block.ancestors().len(), 3, "A4 should have 3 ancestors (excluded D3)");

        // Verify D3 is NOT among A4's ancestors
        let slot_d3 = Slot::new(3, AuthorityIndex::new_for_test(3));
        let d3_blocks = dag_builder.get_uncommitted_blocks_at_slot(slot_d3);
        assert_eq!(d3_blocks.len(), 1);
        let d3_ref = d3_blocks[0].reference();
        assert!(
            !a4_block.ancestors().contains(&d3_ref),
            "A4 should not contain D3 as ancestor"
        );

        // Verify round 5 block C only has A4 as ancestor
        let slot_c5 = Slot::new(5, AuthorityIndex::new_for_test(2));
        let c5_blocks = dag_builder.get_uncommitted_blocks_at_slot(slot_c5);
        assert_eq!(c5_blocks.len(), 1, "Expected exactly one block at C5");
        let c5_block = &c5_blocks[0];
        assert_eq!(c5_block.ancestors().len(), 1, "C5 should have exactly 1 ancestor (A4)");

        let a4_ref = a4_blocks[0].reference();
        assert!(
            c5_block.ancestors().contains(&a4_ref),
            "C5's ancestor should be A4"
        );

        // Verify round 5 block D also only has A4 as ancestor
        let slot_d5 = Slot::new(5, AuthorityIndex::new_for_test(3));
        let d5_blocks = dag_builder.get_uncommitted_blocks_at_slot(slot_d5);
        assert_eq!(d5_blocks.len(), 1, "Expected exactly one block at D5");
        assert_eq!(d5_blocks[0].ancestors().len(), 1, "D5 should have exactly 1 ancestor (A4)");
        assert!(
            d5_blocks[0].ancestors().contains(&a4_ref),
            "D5's ancestor should be A4"
        );
    }

    #[tokio::test]
    async fn test_genesis_round_parsing() {
        let genesis_str = "Round 0 : { 4 }";
        let (_, num_authorities) = parse_genesis(genesis_str).expect("Failed to parse genesis");
        assert_eq!(num_authorities, 4, "Expected 4 authorities from genesis");
    }

    #[tokio::test]
    async fn test_slot_parsing() {
        let slot_str = "A0";
        let (_, slot) = parse_slot(slot_str).expect("Failed to parse slot");
        let expected = Slot::new(0, AuthorityIndex::new_for_test(0));
        assert_eq!(slot, expected, "Expected slot A0");

        let slot_str = "D3";
        let (_, slot) = parse_slot(slot_str).expect("Failed to parse slot");
        let expected = Slot::new(3, AuthorityIndex::new_for_test(3));
        assert_eq!(slot, expected, "Expected slot D3");

        let slot_str = "B12";
        let (_, slot) = parse_slot(slot_str).expect("Failed to parse slot");
        let expected = Slot::new(12, AuthorityIndex::new_for_test(1));
        assert_eq!(slot, expected, "Expected slot B12");
    }

    #[tokio::test]
    async fn test_all_round_parsing() {
        // Build a minimal DAG so we can parse a wildcard round
        let dag_str = "DAG {
            Round 0 : { 4 },
            Round 1 : { * },
        }";

        let (_, dag_builder) = parse_dag(dag_str).expect("Invalid dag");

        // All 4 authorities should have blocks at round 1
        for i in 0..4u32 {
            let slot = Slot::new(1, AuthorityIndex::new_for_test(i));
            let blocks = dag_builder.get_uncommitted_blocks_at_slot(slot);
            assert_eq!(
                blocks.len(),
                1,
                "Expected 1 block at round 1 for authority {}",
                i
            );
            // Each block at round 1 should have 4 ancestors (all genesis blocks)
            assert_eq!(
                blocks[0].ancestors().len(),
                4,
                "Block at round 1 for authority {} should have 4 ancestors",
                i
            );
        }
    }

    #[tokio::test]
    async fn test_specific_round_parsing() {
        let dag_str = "DAG {
            Round 0 : { 4 },
            Round 1 : { * },
            Round 2 : {
                A -> [A1, B1],
                B -> [-C1],
                C -> [*],
                D -> [A1],
            },
        }";

        let (_, dag_builder) = parse_dag(dag_str).expect("Invalid dag");

        // A2 should have exactly 2 ancestors: A1 and B1
        let slot_a2 = Slot::new(2, AuthorityIndex::new_for_test(0));
        let a2_blocks = dag_builder.get_uncommitted_blocks_at_slot(slot_a2);
        assert_eq!(a2_blocks.len(), 1);
        assert_eq!(a2_blocks[0].ancestors().len(), 2, "A2 should have 2 ancestors (A1, B1)");

        // B2 should have 3 ancestors (all of round 1 except C1)
        let slot_b2 = Slot::new(2, AuthorityIndex::new_for_test(1));
        let b2_blocks = dag_builder.get_uncommitted_blocks_at_slot(slot_b2);
        assert_eq!(b2_blocks.len(), 1);
        assert_eq!(b2_blocks[0].ancestors().len(), 3, "B2 should have 3 ancestors (excluded C1)");

        // Verify C1 is not among B2's ancestors
        let slot_c1 = Slot::new(1, AuthorityIndex::new_for_test(2));
        let c1_blocks = dag_builder.get_uncommitted_blocks_at_slot(slot_c1);
        assert_eq!(c1_blocks.len(), 1);
        let c1_ref = c1_blocks[0].reference();
        assert!(
            !b2_blocks[0].ancestors().contains(&c1_ref),
            "B2 should not contain C1 as ancestor"
        );

        // C2 should have 4 ancestors (wildcard = all of round 1)
        let slot_c2 = Slot::new(2, AuthorityIndex::new_for_test(2));
        let c2_blocks = dag_builder.get_uncommitted_blocks_at_slot(slot_c2);
        assert_eq!(c2_blocks.len(), 1);
        assert_eq!(c2_blocks[0].ancestors().len(), 4, "C2 should have 4 ancestors (wildcard)");

        // D2 should have exactly 1 ancestor: A1
        let slot_d2 = Slot::new(2, AuthorityIndex::new_for_test(3));
        let d2_blocks = dag_builder.get_uncommitted_blocks_at_slot(slot_d2);
        assert_eq!(d2_blocks.len(), 1);
        assert_eq!(d2_blocks[0].ancestors().len(), 1, "D2 should have 1 ancestor (A1)");
    }

    #[tokio::test]
    async fn test_parse_author_and_connections() {
        // Test wildcard connection
        let input = "A -> [*],\n";
        let (_, (author, connections)) =
            parse_author_and_connections(input).expect("Failed to parse wildcard connection");
        assert_eq!(author, AuthorityIndex::new_for_test(0));
        assert_eq!(connections, vec!["*"]);

        // Test specific connections
        let input = "B -> [A1, C2],\n";
        let (_, (author, connections)) =
            parse_author_and_connections(input).expect("Failed to parse specific connections");
        assert_eq!(author, AuthorityIndex::new_for_test(1));
        assert_eq!(connections, vec!["A1", "C2"]);

        // Test excluded connection
        let input = "C -> [-D3],\n";
        let (_, (author, connections)) =
            parse_author_and_connections(input).expect("Failed to parse excluded connection");
        assert_eq!(author, AuthorityIndex::new_for_test(2));
        assert_eq!(connections, vec!["-D3"]);

        // Test mixed wildcard and specific
        let input = "D -> [*, A1],\n";
        let (_, (author, connections)) =
            parse_author_and_connections(input).expect("Failed to parse mixed connections");
        assert_eq!(author, AuthorityIndex::new_for_test(3));
        assert_eq!(connections, vec!["*", "A1"]);
    }

    #[tokio::test]
    async fn test_str_to_authority_index() {
        // Single uppercase letters: A=0, B=1, ..., Z=25
        assert_eq!(
            str_to_authority_index("A"),
            Some(AuthorityIndex::new_for_test(0))
        );
        assert_eq!(
            str_to_authority_index("B"),
            Some(AuthorityIndex::new_for_test(1))
        );
        assert_eq!(
            str_to_authority_index("Z"),
            Some(AuthorityIndex::new_for_test(25))
        );

        // Bracketed numeric index for values > 25
        assert_eq!(
            str_to_authority_index("[26]"),
            Some(AuthorityIndex::new_for_test(26))
        );
        assert_eq!(
            str_to_authority_index("[100]"),
            Some(AuthorityIndex::new_for_test(100))
        );
        assert_eq!(
            str_to_authority_index("[0]"),
            Some(AuthorityIndex::new_for_test(0))
        );

        // Invalid inputs
        assert_eq!(str_to_authority_index("a"), None); // lowercase
        assert_eq!(str_to_authority_index("AB"), None); // multi-char non-bracketed
        assert_eq!(str_to_authority_index("[]"), None); // empty brackets
        assert_eq!(str_to_authority_index("[abc]"), None); // non-numeric in brackets
        assert_eq!(str_to_authority_index(""), None); // empty string
    }
}
