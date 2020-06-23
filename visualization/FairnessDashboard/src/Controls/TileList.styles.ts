import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontSizes } from 'office-ui-fabric-react';

export interface ITilesListStyles {
    container: IStyle;
    itemCell: IStyle;
    iconClass: IStyle;
    title: IStyle;
    description: IStyle;
}

export const TileListStyles: () => IProcessedStyleSet<ITilesListStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<ITilesListStyles>({
        container: {
            display: 'inline-flex',
            flexDirection: 'row',
            flexWrap: 'wrap',
            justifyContent: 'space-between',
            borderBottom: '1px solid',
            borderBottomColor: theme.semanticColors.bodyDivider,
        },
        itemCell: {
            padding: '15px',
            width: '235px',
            position: 'relative',
            float: 'left',
            cursor: 'pointer',
            boxSizing: 'border-box',
            backgroundColor: theme.semanticColors.bodyDivider,
            marginBottom: '10px',
            marginRight: '10px',
            selectors: {
                '&:hover': { background: theme.semanticColors.bodyBackgroundHovered },
            },
        },
        iconClass: {
            color: theme.semanticColors.accentButtonBackground,
            position: 'absolute',
            right: '10px',
            top: '10px',
        },
        title: {
            color: theme.semanticColors.bodyText,
            paddingRight: '16px',
            margin: 0,
        },
        description: {
            paddingTop: '10px',
            color: theme.semanticColors.bodySubtext,
        },
    });
};
