import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontSizes, FontWeights } from 'office-ui-fabric-react';

export interface IFeatureTabStyles {
    itemCell: IStyle;
    iconClass: IStyle;
    itemsList: IStyle;
    frame: IStyle;
    main: IStyle;
    header: IStyle;
    textBody: IStyle;
    tableHeader: IStyle;
    itemTitle: IStyle;
    valueCount: IStyle;
    iconWrapper: IStyle;
    featureDescriptionSection: IStyle;
    binSection: IStyle;
    expandButton: IStyle;
    category: IStyle;
    subgroupHeader: IStyle;
}

export const FeatureTabStyles: () => IProcessedStyleSet<IFeatureTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IFeatureTabStyles>({
        itemCell: {
            display: 'flex',
            flexDirection: 'row',
            padding: '20px 0',
            width: '100%',
            cursor: 'pointer',
            boxSizing: 'border-box',
            borderBottom: '1px solid',
            borderBottomColor: theme.semanticColors.bodyDivider,
            selectors: {
                '&:hover': { background: theme.semanticColors.listItemBackgroundHovered },
            },
        },
        iconClass: {
            fontSize: FontSizes.large,
            color: theme.semanticColors.accentButtonBackground,
        },
        itemsList: {
            overflowY: 'auto',
        },
        frame: {
            height: '100%',
        },
        main: {
            height: '100%',
            maxWidth: '700px',
            flex: 1,
        },
        header: {
            color: theme.semanticColors.bodyText,
            fontWeight: FontWeights.semibold,
            margin: '26px 0',
        },
        textBody: {
            paddingTop: '12px',
            fontWeight: FontWeights.semilight,
            color: theme.semanticColors.bodyText,
            paddingBottom: '12px',
        },
        tableHeader: {
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
            paddingBottom: '15px',
            color: theme.semanticColors.bodyText,
            borderBottom: '1px solid',
            borderColor: theme.semanticColors.bodyDivider,
        },
        itemTitle: {
            margin: 0,
            color: theme.semanticColors.listText,
        },
        valueCount: {
            paddingTop: '15px',
            color: theme.semanticColors.bodyText,
        },
        iconWrapper: {
            paddingTop: '4px',
            paddingLeft: '5px',
            width: '30px',
        },
        featureDescriptionSection: {
            flex: 1,
            paddingRight: '20px',
            minHeight: '75px',
        },
        binSection: {
            width: '130px',
        },
        expandButton: {
            paddingLeft: 0,
            selectors: {
                '& i': {
                    marginLeft: 0,
                },
            },
        },
        category: {
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            color: theme.semanticColors.bodyText,
        },
        subgroupHeader: {
            width: '130px',
        },
    });
};
