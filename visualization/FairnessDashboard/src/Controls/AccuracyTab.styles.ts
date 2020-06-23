import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights, FontSizes } from 'office-ui-fabric-react';

export interface IAccuracyTabStyles {
    iconClass: IStyle;
    itemsList: IStyle;
    frame: IStyle;
    main: IStyle;
    header: IStyle;
    textBody: IStyle;
}

export const AccuracyTabStyles: () => IProcessedStyleSet<IAccuracyTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IAccuracyTabStyles>({
        iconClass: {
            fontSize: '20px',
            position: 'absolute',
            right: '10px',
            top: '10px',
        },
        itemsList: {
            overflowY: 'auto',
        },
        frame: {
            height: '100%',
        },
        main: {
            height: '100%',
            maxWidth: '750px',
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
    });
};
