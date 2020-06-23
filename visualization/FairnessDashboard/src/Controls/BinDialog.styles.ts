import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights, FontSizes } from 'office-ui-fabric-react';

export interface IBinDialogStyles {
    frame: IStyle;
    header: IStyle;
    buttons: IStyle;
    saveButton: IStyle;
    cancelButton: IStyle;
    binCounter: IStyle;
    main: IStyle;
    controls: IStyle;
    scrollArea: IStyle;
    groupLabel: IStyle;
}

export const BinDialogStyles: () => IProcessedStyleSet<IBinDialogStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IBinDialogStyles>({
        frame: {
            height: '400px',
            width: '500px',
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: theme.semanticColors.bodyBackground,
        },
        header: {
            padding: '12px',
            textAlign: 'center',
            backgroundColor: theme.semanticColors.bodyFrameBackground,
            color: theme.semanticColors.bodyText,
        },
        buttons: {
            display: 'inline-flex',
            flexDirection: 'row-reverse',
            padding: '10px',
        },
        saveButton: {
            height: '44px',
            padding: '12px',
            marginLeft: '10px',
        },
        cancelButton: {
            height: '44px',
            padding: '12px',
        },
        binCounter: {
            selectors: {
                '& label': {
                    color: theme.semanticColors.bodyText,
                    fontSize: FontSizes.mediumPlus,
                    fontWeight: FontWeights.regular,
                },
            },
        },
        main: {
            flexGrow: 1,
            padding: '20px 40px',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
        },
        controls: {
            display: 'inline-flex',
            width: '100%',
            justifyContent: 'space-between',
            height: '30px',
            alignItems: 'center',
            paddingBottom: '10px',
            borderBottom: '1px solid',
            borderBottomColor: theme.semanticColors.bodyDivider,
            marginBottom: '10px',
        },
        scrollArea: {
            overflowY: 'auto',
            overflowX: 'hidden',
            flexGrow: '2',
        },
        groupLabel: {
            color: theme.semanticColors.bodyText,
            height: '25px',
            borderBottom: '1px solid',
            borderBottomColor: theme.semanticColors.bodyDivider,
            paddingLeft: '12px',
        },
    });
};
