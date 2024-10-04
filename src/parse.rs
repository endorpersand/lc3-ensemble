//! Parsing assembly source code into an AST.
//! 
//! This module is used to convert strings (which represent assembly source code)
//! into abstract syntax trees that maintain all of the information of the source code
//! in an easier to handle format.
//! 
//! The main function to use from this module is [`parse_ast`], 
//! which parses an assembly code program into an AST.
//! 
//! However, if needed, the internals of this module are also available:
//! - [`lex`]: the implementation of the lexer/tokenizer
//! - [`Parser`]: the main logic for the parser
//! - [`Parse`]: the implementation to "parse" an AST component

pub mod lex;

use std::borrow::Cow;
use std::ops::Range;

use logos::{Logos, Span};

use crate::ast::asm::{AsmInstr, Directive, Stmt, StmtKind};
use crate::ast::{ImmOrReg, Offset, OffsetNewErr, PCOffset};
use lex::{Ident, Token};
use simple::*;

use self::lex::LexErr;

/// Parses an assembly source code string into a `Vec` of statements.
/// 
/// # Example
/// ```
/// use lc3_ensemble::parse::parse_ast;
/// 
/// let src = "
///     .orig x3000
///     THIS: ADD R0, R0, #0
///     IS: ADD R1, R1, #1
///     A: ADD R2, R2, #2
///     PROGRAM: ADD R3, R3, #3
///     .end
/// ";
/// 
/// let ast = parse_ast(src).unwrap();
/// assert_eq!(ast.len(), 6);
/// ```
pub fn parse_ast(s: &str) -> Result<Vec<Stmt>, ParseErr> {
    let mut parser = Parser::new(s)?;
    // Horrendous one-liner version of this:
    // std::iter::from_fn(|| (!parser.is_empty()).then(|| parser.parse())).collect()
    std::iter::from_fn(|| match parser.is_empty() {
        true  => None,
        false => Some(parser.parse()),
    }).collect::<Result<Vec<_>, _>>()
}

enum ParseErrKind {
    OffsetNew(OffsetNewErr),
    Lex(LexErr),
    Parse(Cow<'static, str>)
}
impl From<LexErr> for ParseErrKind {
    fn from(value: LexErr) -> Self {
        Self::Lex(value)
    }
}
impl From<OffsetNewErr> for ParseErrKind {
    fn from(value: OffsetNewErr) -> Self {
        Self::OffsetNew(value)
    }
}
/// Any error that occurs during parsing tokens.
pub struct ParseErr {
    /// The brief cause of this error.
    kind: ParseErrKind,
    /// Some kind of help (if it exists)
    help: Cow<'static, str>,
    /// The location of this error.
    span: Span
}
impl ParseErr {
    fn new<S: Into<Cow<'static, str>>>(msg: S, span: Span) -> Self {
        Self { kind: ParseErrKind::Parse(msg.into()), help: Cow::Borrowed(""), span }
    }

    fn wrap<E: Into<ParseErrKind>>(err: E, span: Span) -> Self {
        Self { kind: err.into(), help: Cow::Borrowed(""), span }
    }

    fn with_help<S: Into<Cow<'static, str>>>(mut self, help: S) -> Self {
        self.help = help.into();
        self
    }
}
impl std::fmt::Debug for ParseErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParseErr")
            .field("brief", match &self.kind {
                ParseErrKind::OffsetNew(s) => s,
                ParseErrKind::Lex(s) => s,
                ParseErrKind::Parse(s) => s,
            })
            .field("span", &self.span)
            .finish()
    }
}
impl std::fmt::Display for ParseErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ParseErrKind::OffsetNew(e) => e.fmt(f),
            ParseErrKind::Lex(e) => e.fmt(f),
            ParseErrKind::Parse(s) => s.fmt(f),
        }
    }
}
impl std::error::Error for ParseErr {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ParseErrKind::OffsetNew(e) => Some(e),
            ParseErrKind::Lex(e) => Some(e),
            ParseErrKind::Parse(_) => None,
        }
    }
}
impl crate::err::Error for ParseErr {
    fn span(&self) -> Option<crate::err::ErrSpan> {
        Some(crate::err::ErrSpan::from(self.span.clone()))
    }
        
    fn help(&self) -> Option<Cow<str>> {
        match &self.kind {
            ParseErrKind::OffsetNew(e) => e.help(),
            ParseErrKind::Lex(e) => e.help(),
            ParseErrKind::Parse(_) => Some(Cow::Borrowed(&self.help)),
        }
    }
}

/// Components that can be constructed from a sequence of tokens.
pub trait Parse: Sized {
    /// Attempt to convert the next sequence of tokens 
    /// in the parser's state into a component.
    /// 
    /// If parsing fails, there are no guarantees about what happens to the input,
    /// and the parser likely should not be used after an error is raised during parsing.
    fn parse(parser: &mut Parser) -> Result<Self, ParseErr>;
}

/// The main parser struct, which holds the main logic for the parser.
pub struct Parser {
    tokens: Vec<(Token, Span)>,
    index: usize,
    spans: Vec<Span>,
}
impl Parser {
    /// Creates a new parser from a given string.
    /// 
    /// In the instantiation process, 
    /// this function will attempt to tokenize the string into tokens,
    /// raising an error if it fails.
    pub fn new(stream: &str) -> Result<Self, ParseErr> {
        let tokens = Token::lexer(stream).spanned()
            .map(|(m_token, span)| match m_token {
                Ok(token) => Ok((token, span)),
                Err(err)  => Err(ParseErr::wrap(err, span)),
            })
            .filter(|t| !matches!(t, Ok((Token::Comment, _)))) // filter comments
            .collect::<Result<_, _>>()?;

        Ok(Self { tokens, index: 0, spans: vec![] })
    }

    /// Peeks at the next token to read.
    pub fn peek(&self) -> Option<&(Token, Span)> {
        self.tokens[self.index..].first()
    }
    /// Advances the parser ahead by one token.
    pub fn advance(&mut self) {
        // Append the last token's span to the last span collector.
        let last_tok_span = self.cursor();
        if let Some(last_span) = self.spans.last_mut() {
            last_span.end = last_tok_span.end;
        }

        self.index += 1;
        self.index = self.index.min(self.tokens.len());
    }
    /// Gets the range of the next token to read (or an EOL range if there are no more tokens to read).
    pub fn cursor(&self) -> Span {
        match self.peek().or_else(|| self.tokens.last()) {
            Some((_, span)) => span.clone(),
            None => 0..0
        }
    }

    /// Parses the current token stream into a component, erroring if not possible.
    /// 
    /// If parsing fails, there are no guarantees about what happens to the input,
    /// and the parser likely should not be used after an error is raised during parsing.
    pub fn parse<P: Parse>(&mut self) -> Result<P, ParseErr> {
        P::parse(self)
    }

    /// Check if the next token matches the given component and consume it if so.
    /// 
    /// This function can error if the next token *does* match the given component,
    /// but an error occurs in trying to convert it to that component.
    pub fn match_<P: TokenParse>(&mut self) -> Result<Option<P>, ParseErr> {
        let span = self.cursor();
        match self.advance_if(P::match_) {
            Ok(t)  => P::convert(t, span).map(Some),
            Err(_) => Ok(None),
        }
    }

    /// Applies the provided predicate to the next token in the input.
    /// 
    /// If an error is raised from the predicate, the parser does not advance its input.
    pub fn advance_if<T>(&mut self, pred: impl FnOnce(Option<&Token>, Span) -> Result<T, ParseErr>) -> Result<T, ParseErr> {
        let result = if let Some((tok, span)) = self.peek() {
            pred(Some(tok), span.clone())
        } else {
            pred(None, self.cursor())
        };
        if result.is_ok() {
            self.advance();
        }
        result
    }

    /// Calculates the span of the component created inside this region block.
    pub fn spanned<T, E>(&mut self, f: impl FnOnce(&mut Parser) -> Result<T, E>) -> Result<(T, Range<usize>), E> {
        let Range { start, end: _ } = self.cursor();
        
        self.spans.push(start..start);
        let result = f(self);

        // pop span
        let span = self.spans.pop().unwrap();
        if let Some(last_span) = self.spans.last_mut() {
            last_span.end = span.end;
        }

        Ok((result?, span))
    }

    /// Checks whether the input for the parser is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens[self.index..]
            .iter()
            .all(|(t, _)| t.is_whitespace())
    }
}

impl<const N: u32> Parse for ImmOrReg<N> {
    fn parse(parser: &mut Parser) -> Result<Self, ParseErr> {
        match parser.match_()? {
            Some(Either::Left(imm))  => Ok(ImmOrReg::Imm(imm)),
            Some(Either::Right(reg)) => Ok(ImmOrReg::Reg(reg)),
            None => Err(ParseErr::new("expected register or immediate value", parser.cursor()))
        }
    }
}

impl<OFF, const N: u32> Parse for PCOffset<OFF, N> 
    where Offset<OFF, N>: TokenParse
{
    fn parse(parser: &mut Parser) -> Result<Self, ParseErr> {
        match parser.match_()? {
            Some(Either::Left(off)) => Ok(PCOffset::Offset(off)),
            Some(Either::Right(label)) => Ok(PCOffset::Label(label)),
            None => Err(ParseErr::new("expected offset or label", parser.cursor()))
        }
    }
}

/// Simple to parse components.
/// 
/// This module holds components that are very simple to parse
/// (defined as only requiring a single token and no additional state from the parser).
/// 
/// The key type of this module is the [`TokenParse`] trait which defines
/// how to "simply parse" a component. 
/// See that trait for more details about its utility over [`Parse`].
/// 
/// This module also provides several utility parsers (e.g., [`Comma`] and [`Colon`])
/// for use in more complex component parsing.
pub mod simple {
    use logos::Span;

    use crate::ast::{Label, Offset, Reg};

    use super::lex::{Ident, LexErr, Token};
    use super::{Parse, ParseErr, Parser};

    /// Components that can be constructed with a single token 
    /// and require no additional parser state.
    /// 
    /// This has an advantage over [`Parse`] in that if parsing fails,
    /// the parser is known to not advance its input. 
    /// This can be taken advantage of with [`Parser::match_`], 
    /// which only advances if parsing passes.
    /// 
    /// [`Parser::match_`]: super::Parser::match_
    pub trait TokenParse: Sized {
        /// An intermediate to hold the match before it is converted to the actual component.
        type Intermediate;

        /// Tries to match the next token to the given component, if possible.
        /// 
        /// If successful, this returns some value and the parser advances. 
        /// If unsuccessful, this returns an error and the parser does not advance.
        /// 
        /// The value returned is an intermediate value which is later converted to the desired component.
        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self::Intermediate, ParseErr>;

        /// Parses the intermediate into the given component, raising an error if conversion fails.
        fn convert(imed: Self::Intermediate, span: Span) -> Result<Self, ParseErr>;
    }
    impl<S: TokenParse> Parse for S {
        fn parse(parser: &mut Parser) -> Result<Self, ParseErr> {
            let span = parser.cursor();
            let imed = parser.advance_if(S::match_)?;
            S::convert(imed, span)
        }
    }
    trait DirectTokenParse: TokenParse<Intermediate = Self> {
        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self, ParseErr>;
    }
    impl<T: DirectTokenParse> TokenParse for T {
        type Intermediate = Self;
    
        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self::Intermediate, ParseErr> {
            DirectTokenParse::match_(m_token, span)
        }
    
        fn convert(imed: Self::Intermediate, _span: Span) -> Result<Self, ParseErr> {
            Ok(imed)
        }
    }

    /// Comma.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
    pub struct Comma;
    impl DirectTokenParse for Comma {
        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self::Intermediate, ParseErr> {
            match m_token {
                Some(Token::Comma) => Ok(Comma),
                _ => Err(ParseErr::new("expected comma", span))
            }
        }
    }

    /// Colon.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
    pub struct Colon;
    impl DirectTokenParse for Colon {
        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self, ParseErr> {
            match m_token {
                Some(Token::Colon) => Ok(Colon),
                _ => Err(ParseErr::new("expected colon", span))
            }
        }
    }

    /// A string literal.
    #[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
    pub struct StrLiteral(pub String);
    impl DirectTokenParse for StrLiteral {
        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self, ParseErr> {
            match m_token {
                Some(Token::String(s)) => Ok(StrLiteral(s.to_string())),
                _ => Err(ParseErr::new("expected string literal", span))
            }
        }
    }

    /// The end of a line or input.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
    pub struct End;
    impl DirectTokenParse for End {
        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self, ParseErr> {
            match m_token {
                None | Some(Token::NewLine) => Ok(End),
                _ => Err(ParseErr::new("expected end of line", span))
            }
        }
    }

    /// An (signed or unsigned) int literal. 
    /// This is primarily only used for `.fill`, which is sign-agnostic.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
    pub struct IntLiteral(pub u16);
    impl DirectTokenParse for IntLiteral {
        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self, ParseErr> {
            match m_token {
                Some(&Token::Unsigned(n)) => Ok(Self(n)),
                Some(&Token::Signed(n)) => Ok(Self(n as u16)),
                _ => Err(ParseErr::new("expected immediate value", span.clone()))
            }
        }
    }

    /// Either one component or another.
    /// 
    /// This is not meant to be used as a general purpose Either type.
    /// It is only meant to be used for parsing.
    pub enum Either<L, R> {
        /// The first possible component.
        Left(L),
        /// The second possible component.
        Right(R)
    }
    impl<L: TokenParse, R: TokenParse> TokenParse for Either<L, R> {
        type Intermediate = Either<L::Intermediate, R::Intermediate>;
        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self::Intermediate, ParseErr> {
            match L::match_(m_token, span.clone()) {
                Ok(t) => Ok(Either::Left(t)),
                Err(_) => match R::match_(m_token, span.clone()) {
                    Ok(u) => Ok(Either::Right(u)),
                    Err(_) => Err(ParseErr::new("could not parse", span)),
                },
            }
        }
        
        fn convert(imed: Self::Intermediate, span: Span) -> Result<Self, ParseErr> {
            match imed {
                Either::Left(l)  => L::convert(l, span).map(Either::Left),
                Either::Right(r) => R::convert(r, span).map(Either::Right),
            }
        }
    }

    impl DirectTokenParse for Reg {
        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self, ParseErr> {
            match m_token {
                Some(&Token::Reg(reg_no)) => Reg::try_from(reg_no)
                    .map_err(|_| ParseErr::new(format!("invalid register number {reg_no}"), span)),
                _ => Err(ParseErr::new("expected register", span))
            }
        }
    }

    impl<const N: u32> TokenParse for Offset<i16, N> {
        type Intermediate = Either<i16, u16>;

        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self::Intermediate, ParseErr> {
            match m_token {
                Some(&Token::Unsigned(n)) => Ok(Either::Right(n)),
                Some(&Token::Signed(n))   => Ok(Either::Left(n)),
                _ => Err(ParseErr::new("expected immediate value", span.clone()))
            }
        }
        
        fn convert(imed: Self::Intermediate, span: Span) -> Result<Self, ParseErr> {
            let off_val = match imed {
                Either::Left(n)  => n,
                Either::Right(n) => {
                    <_>::try_from(n).map_err(|_| ParseErr::wrap(LexErr::DoesNotFitI16, span.clone()))?
                },
            };
            
            Self::new(off_val)
                .map_err(|s| ParseErr::wrap(s, span))
        }
    }

    impl<const N: u32> TokenParse for Offset<u16, N> {
        type Intermediate = Either<u16, i16>;

        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self::Intermediate, ParseErr> {
            match m_token {
                Some(&Token::Unsigned(n)) => Ok(Either::Left(n)),
                Some(&Token::Signed(n))   => Ok(Either::Right(n)),
                _ => Err(ParseErr::new("expected immediate value", span.clone()))
            }
        }
        
        fn convert(imed: Self::Intermediate, span: Span) -> Result<Self, ParseErr> {
            let off_val = match imed {
                Either::Left(n)  => n,
                Either::Right(n) => {
                    <_>::try_from(n).map_err(|_| ParseErr::wrap(LexErr::DoesNotFitU16, span.clone()))?
                },
            };
            
            Self::new(off_val)
                .map_err(|s| ParseErr::wrap(s, span))
        }
    }
    impl DirectTokenParse for Label {
        fn match_(m_token: Option<&Token>, span: Span) -> Result<Self, ParseErr> {
            match m_token {
                Some(Token::Ident(Ident::Label(s))) => Ok(Label::new(s.to_string(), span)),
                _ => Err(ParseErr::new("expected label", span))
            }
        }
    }
}

impl Parse for AsmInstr {
    fn parse(parser: &mut Parser) -> Result<Self, ParseErr> {
        let opcode = parser.advance_if(|mt, span| match mt {
            Some(Token::Ident(id)) if !matches!(id, Ident::Label(_)) => Ok(id.clone()),
            _ => Err(ParseErr::new("expected instruction", span))
        })?;

        match opcode {
            Ident::ADD => {
                let dr = parser.parse()?;
                parser.parse::<Comma>()?;
                let sr1 = parser.parse()?;
                parser.parse::<Comma>()?;
                let sr2 = parser.parse()?;

                Ok(Self::ADD(dr, sr1, sr2))
            },
            Ident::AND => {
                let dr = parser.parse()?;
                parser.parse::<Comma>()?;
                let sr1 = parser.parse()?;
                parser.parse::<Comma>()?;
                let sr2 = parser.parse()?;

                Ok(Self::AND(dr, sr1, sr2))
            },
            Ident::BR => Ok(Self::BR(0b111, parser.parse()?)),
            Ident::BRP => Ok(Self::BR(0b001, parser.parse()?)),
            Ident::BRZ => Ok(Self::BR(0b010, parser.parse()?)),
            Ident::BRZP => Ok(Self::BR(0b011, parser.parse()?)),
            Ident::BRN => Ok(Self::BR(0b100, parser.parse()?)),
            Ident::BRNP => Ok(Self::BR(0b101, parser.parse()?)),
            Ident::BRNZ => Ok(Self::BR(0b110, parser.parse()?)),
            Ident::BRNZP => Ok(Self::BR(0b111, parser.parse()?)),
            Ident::JMP => Ok(Self::JMP(parser.parse()?)),
            Ident::JSR => Ok(Self::JSR(parser.parse()?)),
            Ident::JSRR => Ok(Self::JSRR(parser.parse()?)),
            Ident::LD => {
                let dr = parser.parse()?;
                parser.parse::<Comma>()?;
                let off = parser.parse()?;

                Ok(Self::LD(dr, off))
            },
            Ident::LDI => {
                let dr = parser.parse()?;
                parser.parse::<Comma>()?;
                let off = parser.parse()?;

                Ok(Self::LDI(dr, off))
            },
            Ident::LDR => {
                let dr = parser.parse()?;
                parser.parse::<Comma>()?;
                let br = parser.parse()?;
                parser.parse::<Comma>()?;
                let off = parser.parse()?;

                Ok(Self::LDR(dr, br, off))
            },
            Ident::LEA => {
                let dr = parser.parse()?;
                parser.parse::<Comma>()?;
                let off = parser.parse()?;

                Ok(Self::LEA(dr, off))
            },
            Ident::NOT => {
                let dr = parser.parse()?;
                parser.parse::<Comma>()?;
                let sr = parser.parse()?;

                Ok(Self::NOT(dr, sr))
            },
            Ident::RET => Ok(Self::RET),
            Ident::RTI => Ok(Self::RTI),
            Ident::ST => {
                let sr = parser.parse()?;
                parser.parse::<Comma>()?;
                let off = parser.parse()?;

                Ok(Self::ST(sr, off))
            },
            Ident::STI => {
                let sr = parser.parse()?;
                parser.parse::<Comma>()?;
                let off = parser.parse()?;

                Ok(Self::STI(sr, off))
            },
            Ident::STR => {
                let dr = parser.parse()?;
                parser.parse::<Comma>()?;
                let br = parser.parse()?;
                parser.parse::<Comma>()?;
                let off = parser.parse()?;

                Ok(Self::STR(dr, br, off))
            },
            Ident::TRAP => Ok(Self::TRAP(parser.parse()?)),
            Ident::NOP => {
                // NOP can optionally accept a parameter.
                let off = match parser.peek() {
                    Some((Token::Signed(_) | Token::Unsigned(_) | Token::Ident(Ident::Label(_)), _)) => parser.parse()?,
                    _ => PCOffset::Offset(Offset::new_trunc(0)),
                };

                Ok(Self::NOP(off))
            },
            Ident::GETC => Ok(Self::GETC),
            Ident::OUT => Ok(Self::OUT),
            Ident::PUTC => Ok(Self::PUTC),
            Ident::PUTS => Ok(Self::PUTS),
            Ident::IN => Ok(Self::IN),
            Ident::PUTSP => Ok(Self::PUTSP),
            Ident::HALT => Ok(Self::HALT),
            Ident::Label(_) => Err(ParseErr::new("expected instruction", parser.cursor())) // should be unreachable
        }
    }
}

impl Parse for Directive {
    fn parse(parser: &mut Parser) -> Result<Self, ParseErr> {
        use Either::*;

        let cursor = parser.cursor();
        let directive = parser.advance_if(|mt, span| match mt {
            Some(Token::Directive(id)) => Ok(id.to_string()),
            _ => Err(ParseErr::new("expected directive", span))
        })?;

        match &*directive.to_uppercase() {
            "ORIG" => Ok(Self::Orig(parser.parse()?)),
            "FILL" => {
                // .fill is weird.
                //
                // Unlike other numeric operands, it can accept both unsigned and signed literals,
                // so it cannot be parsed with PCOffset's parser and has to be handled differently.
                let span = parser.cursor();
                let operand = match parser.match_()? {
                    Some(Left(label))            => Ok(PCOffset::Label(label)),
                    Some(Right(IntLiteral(off))) => Ok(PCOffset::Offset(Offset::new_trunc(off))),
                    _ => Err(ParseErr::new("expected numeric or label", span))
                }?;

                Ok(Self::Fill(operand))
            }
            "BLKW" => {
                let span = parser.cursor();
                let block_size: Offset<_, 16> = parser.parse()?;
                match block_size.get() != 0 {
                    true  => Ok(Self::Blkw(block_size)),
                    false => Err(ParseErr::new("block size must be greater than 0", span))
                }
            }
            "STRINGZ" => {
                let StrLiteral(s) = parser.parse()?;
                Ok(Self::Stringz(s))
            }
            "END" => Ok(Self::End),
            _ => Err({
                ParseErr::new("invalid directive", cursor)
                    .with_help("the valid directives are .orig, .fill, .blkw, .stringz, .end")
            })
        }
    }
}

impl Parse for StmtKind {
    fn parse(parser: &mut Parser) -> Result<Self, ParseErr> {
        // This parser exists for consistency, but is not actually used.
        // See it used in the implementation of nucleus in Stmt.
        match parser.peek() {
            Some((Token::Directive(_), _)) => Ok(StmtKind::Directive(parser.parse()?)),
            Some((Token::Ident(id), _)) if !matches!(id, Ident::Label(_)) => Ok(StmtKind::Instr(parser.parse()?)),
            _ => Err(ParseErr::new("expected instruction or directive", parser.cursor()))
        }
    }
}
impl Parse for Stmt {
    fn parse(parser: &mut Parser) -> Result<Self, ParseErr> {
        let mut labels = vec![];

        // gets the span of the last token
        // useful for better error messages
        let mut last_label_span = None;

        // Scan through labels and new lines until we find an instruction
        while !parser.is_empty() {
            let span = parser.cursor();
            match parser.match_()? {
                Some(Either::Left(label)) => {
                    parser.match_::<Colon>()?; // skip colon if it exists

                    last_label_span.replace(span.clone());
                    labels.push(label);
                }
                Some(Either::Right(End)) => {},
                _ => break
            }
        }
        
        let (nucleus, span) = parser.spanned(|parser| {
            match parser.peek() {
                Some((Token::Directive(_), _)) => Ok(StmtKind::Directive(parser.parse()?)),
                Some((Token::Ident(id), _)) if !matches!(id, Ident::Label(_)) => Ok(StmtKind::Instr(parser.parse()?)),
                _ => {
                    // Parser didn't find a directive or instruction following a label.
                    // Chances are the label was just a misspelled instruction.
                    Err(ParseErr::new("expected instruction or directive", last_label_span.unwrap_or(parser.cursor())))
                }
            }
        })?;

        // assert end of line at end of statement
        parser.parse::<End>()?;
        // consume any extra NLs
        while !parser.is_empty() && parser.match_::<End>()?.is_some() {}

        Ok(Self { labels, nucleus, span })
    }
}